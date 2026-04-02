# pypto.program: Qwen332BTrainingForwardBackward
import pypto.language as pl

@pl.program
class Qwen332BTrainingForwardBackward:
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_32b_training_forward_and_backward_layer_incore_0(self, grad_w_down__ssa_v0: pl.Tensor[[400, 80], pl.FP32], grad_w_gate__ssa_v0: pl.Tensor[[80, 400], pl.FP32], grad_w_up__ssa_v0: pl.Tensor[[80, 400], pl.FP32], grad_wk__ssa_v0: pl.Tensor[[80, 80], pl.FP32], grad_wo__ssa_v0: pl.Tensor[[80, 80], pl.FP32], grad_wq__ssa_v0: pl.Tensor[[80, 80], pl.FP32], grad_wv__ssa_v0: pl.Tensor[[80, 80], pl.FP32], ret0__out: pl.Out[pl.Tensor[[400, 80], pl.FP32]], ret1__out: pl.Out[pl.Tensor[[80, 400], pl.FP32]], ret2__out: pl.Out[pl.Tensor[[80, 400], pl.FP32]], ret3__out: pl.Out[pl.Tensor[[80, 80], pl.FP32]], ret4__out: pl.Out[pl.Tensor[[80, 80], pl.FP32]], ret5__out: pl.Out[pl.Tensor[[80, 80], pl.FP32]], ret6__out: pl.Out[pl.Tensor[[80, 80], pl.FP32]]) -> tuple[pl.Tensor[[400, 80], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32]]:
        grad_w_down__tile: pl.Tile[[400, 80], pl.FP32, pl.Mem.Vec] = pl.tile.load(grad_w_down__ssa_v0, [0, 0], [400, 80], [400, 80], target_memory=pl.Mem.Vec, transpose=False)
        grad_w_gate__tile: pl.Tile[[80, 400], pl.FP32, pl.Mem.Vec] = pl.tile.load(grad_w_gate__ssa_v0, [0, 0], [80, 400], [80, 400], target_memory=pl.Mem.Vec, transpose=False)
        grad_w_up__tile: pl.Tile[[80, 400], pl.FP32, pl.Mem.Vec] = pl.tile.load(grad_w_up__ssa_v0, [0, 0], [80, 400], [80, 400], target_memory=pl.Mem.Vec, transpose=False)
        grad_wk__tile: pl.Tile[[80, 80], pl.FP32, pl.Mem.Vec] = pl.tile.load(grad_wk__ssa_v0, [0, 0], [80, 80], [80, 80], target_memory=pl.Mem.Vec, transpose=False)
        grad_wo__tile: pl.Tile[[80, 80], pl.FP32, pl.Mem.Vec] = pl.tile.load(grad_wo__ssa_v0, [0, 0], [80, 80], [80, 80], target_memory=pl.Mem.Vec, transpose=False)
        grad_wq__tile: pl.Tile[[80, 80], pl.FP32, pl.Mem.Vec] = pl.tile.load(grad_wq__ssa_v0, [0, 0], [80, 80], [80, 80], target_memory=pl.Mem.Vec, transpose=False)
        grad_wv__tile: pl.Tile[[80, 80], pl.FP32, pl.Mem.Vec] = pl.tile.load(grad_wv__ssa_v0, [0, 0], [80, 80], [80, 80], target_memory=pl.Mem.Vec, transpose=False)
        grad_wq__tile_1: pl.Tile[[80, 80], pl.FP32, pl.Mem.Vec] = pl.tile.muls(grad_wq__tile, 0.0)
        grad_wk__tile_1: pl.Tile[[80, 80], pl.FP32, pl.Mem.Vec] = pl.tile.muls(grad_wk__tile, 0.0)
        grad_wv__tile_1: pl.Tile[[80, 80], pl.FP32, pl.Mem.Vec] = pl.tile.muls(grad_wv__tile, 0.0)
        grad_wo__tile_1: pl.Tile[[80, 80], pl.FP32, pl.Mem.Vec] = pl.tile.muls(grad_wo__tile, 0.0)
        grad_w_gate__tile_1: pl.Tile[[80, 400], pl.FP32, pl.Mem.Vec] = pl.tile.muls(grad_w_gate__tile, 0.0)
        grad_w_up__tile_1: pl.Tile[[80, 400], pl.FP32, pl.Mem.Vec] = pl.tile.muls(grad_w_up__tile, 0.0)
        grad_w_down__tile_1: pl.Tile[[400, 80], pl.FP32, pl.Mem.Vec] = pl.tile.muls(grad_w_down__tile, 0.0)
        ret0__store: pl.Tensor[[400, 80], pl.FP32] = pl.tile.store(grad_w_down__tile_1, [0, 0], ret0__out)
        ret1__store: pl.Tensor[[80, 400], pl.FP32] = pl.tile.store(grad_w_gate__tile_1, [0, 0], ret1__out)
        ret2__store: pl.Tensor[[80, 400], pl.FP32] = pl.tile.store(grad_w_up__tile_1, [0, 0], ret2__out)
        ret3__store: pl.Tensor[[80, 80], pl.FP32] = pl.tile.store(grad_wk__tile_1, [0, 0], ret3__out)
        ret4__store: pl.Tensor[[80, 80], pl.FP32] = pl.tile.store(grad_wo__tile_1, [0, 0], ret4__out)
        ret5__store: pl.Tensor[[80, 80], pl.FP32] = pl.tile.store(grad_wq__tile_1, [0, 0], ret5__out)
        ret6__store: pl.Tensor[[80, 80], pl.FP32] = pl.tile.store(grad_wv__tile_1, [0, 0], ret6__out)
        return ret0__store, ret1__store, ret2__store, ret3__store, ret4__store, ret5__store, ret6__store
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_32b_training_forward_and_backward_layer_incore_1(self, loss_acc__ssa_v0: pl.Tensor[[2, 1], pl.FP32], ret0__out: pl.Out[pl.Tensor[[2, 1], pl.FP32]]) -> pl.Tensor[[2, 1], pl.FP32]:
        loss_acc__tile: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.load(loss_acc__ssa_v0, [0, 0], [2, 1], [2, 1], target_memory=pl.Mem.Vec, transpose=False)
        loss_acc__tile_1: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(loss_acc__tile, 0.0)
        ret0__store: pl.Tensor[[2, 1], pl.FP32] = pl.tile.store(loss_acc__tile_1, [0, 0], ret0__out)
        return ret0__store
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_32b_training_forward_and_backward_layer_incore_2(self, btrans_scratch__ssa_v0: pl.InOut[pl.Tensor[[2, 4], pl.FP32]], hidden_states__ssa_v0: pl.Tensor[[1, 4, 80], pl.BF16], input_rms_weight__ssa_v0: pl.Tensor[[1, 80], pl.FP32], loss_acc__ssa_v1: pl.InOut[pl.Tensor[[2, 1], pl.FP32]], out__ssa_v0: pl.Out[pl.Tensor[[1, 4, 80], pl.BF16]], post_rms_weight__ssa_v0: pl.Tensor[[1, 80], pl.FP32], target_states__ssa_v0: pl.Tensor[[1, 4, 80], pl.BF16], tok_blocks__ssa_v0: pl.Scalar[pl.INDEX], w_down__ssa_v0: pl.Tensor[[400, 80], pl.BF16], w_gate__ssa_v0: pl.Tensor[[80, 400], pl.BF16], w_up__ssa_v0: pl.Tensor[[80, 400], pl.BF16], wk__ssa_v0: pl.Tensor[[80, 80], pl.BF16], wo__ssa_v0: pl.Tensor[[80, 80], pl.BF16], wq__ssa_v0: pl.Tensor[[80, 80], pl.BF16], wv__ssa_v0: pl.Tensor[[80, 80], pl.BF16]) -> tuple[pl.Tensor[[2, 1], pl.FP32], pl.Tensor[[1, 4, 80], pl.BF16]]:
        for b__cr_idx_v0, (btrans_scratch__cr_iter_v1, loss_acc__cr_iter_v2, out__cr_iter_v1) in pl.parallel(1, init_values=(btrans_scratch__ssa_v0, loss_acc__ssa_v1, out__ssa_v0)):
            for p0_idx__idx_v0, (btrans_scratch__iter_v3, loss_acc__iter_v4, out__iter_v3) in pl.range(tok_blocks__ssa_v0, init_values=(btrans_scratch__cr_iter_v1, loss_acc__cr_iter_v2, out__cr_iter_v1)):
                p0__ssa_v0: pl.Scalar[pl.INDEX] = p0_idx__idx_v0 * 2
                sq_sum__tile: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.create([2, 1], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                sq_sum__tile_1: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(sq_sum__tile, 0.0)
                for kb__idx_v0, (sq_sum__iter_v2,) in pl.range(20, init_values=(sq_sum__tile_1,)):
                    k0__ssa_v0: pl.Scalar[pl.INDEX] = kb__idx_v0 * 4
                    t__tile: pl.Tile[[2, 4], pl.BF16, pl.Mem.Vec] = pl.tile.load(hidden_states__ssa_v0, [0 + b__cr_idx_v0 * 1, p0__ssa_v0, k0__ssa_v0], [1, 2, 4], [1, 2, 4], target_memory=pl.Mem.Vec, transpose=False)
                    t__tile_1: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode='round')
                    x_chunk__tile: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [2, 4])
                    t__tile_2: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.mul(x_chunk__tile, x_chunk__tile)
                    tmp_tile: pl.Tile[[2, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    t__tile_3: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.row_sum(t__tile_2, tmp_tile)
                    sq_sum__tile_2: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.add(sq_sum__iter_v2, t__tile_3)
                    sq_sum__rv_v3: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.yield_(sq_sum__tile_2)
                t__tile_4: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(sq_sum__rv_v3, 0.0125)
                t__tile_5: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.adds(t__tile_4, 1e-06)
                inv_rms__tile: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.rsqrt(t__tile_5)
                normed_tile__tile: pl.Tile[[2, 80], pl.BF16, pl.Mem.Vec] = pl.tile.create([2, 80], dtype=pl.BF16, target_memory=pl.Mem.Vec)
                for kb__idx_v0_1, (normed_tile__iter_v1,) in pl.range(20, init_values=(normed_tile__tile,)):
                    k0__ssa_v1: pl.Scalar[pl.INDEX] = kb__idx_v0_1 * 4
                    t__tile_6: pl.Tile[[2, 4], pl.BF16, pl.Mem.Vec] = pl.tile.load(hidden_states__ssa_v0, [0 + b__cr_idx_v0 * 1, p0__ssa_v0, k0__ssa_v1], [1, 2, 4], [1, 2, 4], target_memory=pl.Mem.Vec, transpose=False)
                    t__tile_7: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.cast(t__tile_6, target_type=pl.FP32, mode='round')
                    x_chunk__tile_1: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_7, [2, 4])
                    gamma__tile: pl.Tile[[1, 4], pl.FP32, pl.Mem.Vec] = pl.tile.load(input_rms_weight__ssa_v0, [0, k0__ssa_v1], [1, 4], [1, 4], target_memory=pl.Mem.Vec, transpose=False)
                    t__tile_8: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(x_chunk__tile_1, inv_rms__tile)
                    normed__tile: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_8, gamma__tile)
                    t__tile_9: pl.Tile[[2, 4], pl.BF16, pl.Mem.Vec] = pl.tile.cast(normed__tile, target_type=pl.BF16, mode='round')
                    normed_tile__tile_1: pl.Tile[[2, 80], pl.BF16, pl.Mem.Vec] = pl.tile.assemble(normed_tile__iter_v1, t__tile_9, [0, k0__ssa_v1])
                    normed_tile__rv_v2: pl.Tile[[2, 80], pl.BF16, pl.Mem.Vec] = pl.yield_(normed_tile__tile_1)
                q_proj_tile__tile: pl.Tile[[2, 80], pl.BF16, pl.Mem.Vec] = pl.tile.create([2, 80], dtype=pl.BF16, target_memory=pl.Mem.Vec)
                k_proj_tile__tile: pl.Tile[[2, 80], pl.BF16, pl.Mem.Vec] = pl.tile.create([2, 80], dtype=pl.BF16, target_memory=pl.Mem.Vec)
                v_proj_tile__tile: pl.Tile[[2, 80], pl.BF16, pl.Mem.Vec] = pl.tile.create([2, 80], dtype=pl.BF16, target_memory=pl.Mem.Vec)
                for ob__idx_v0, (k_proj_tile__iter_v1, q_proj_tile__iter_v1, v_proj_tile__iter_v1) in pl.range(10, init_values=(k_proj_tile__tile, q_proj_tile__tile, v_proj_tile__tile)):
                    q0__ssa_v0: pl.Scalar[pl.INDEX] = ob__idx_v0 * 8
                    q_acc__tile: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_acc__tile_1: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(q_acc__tile, 0.0)
                    k_acc__tile: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    k_acc__tile_1: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(k_acc__tile, 0.0)
                    v_acc__tile: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    v_acc__tile_1: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(v_acc__tile, 0.0)
                    for kb__idx_v0_2, (k_acc__iter_v2, q_acc__iter_v2, v_acc__iter_v2) in pl.range(20, init_values=(k_acc__tile_1, q_acc__tile_1, v_acc__tile_1)):
                        k0__ssa_v2: pl.Scalar[pl.INDEX] = kb__idx_v0_2 * 4
                        t__tile_10: pl.Tile[[2, 4], pl.BF16, pl.Mem.Vec] = pl.tile.slice(normed_tile__rv_v2, [2, 4], [0, k0__ssa_v2])
                        n_chunk__tile: pl.Tile[[2, 4], pl.BF16, pl.Mem.Vec] = pl.tile.cast(t__tile_10, target_type=pl.BF16, mode='round')
                        wq_c__tile: pl.Tile[[4, 8], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(wq__ssa_v0, [k0__ssa_v2, q0__ssa_v0], [4, 8], [4, 8], target_memory=pl.Mem.Mat, transpose=False)
                        wk_c__tile: pl.Tile[[4, 8], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(wk__ssa_v0, [k0__ssa_v2, q0__ssa_v0], [4, 8], [4, 8], target_memory=pl.Mem.Mat, transpose=False)
                        wv_c__tile: pl.Tile[[4, 8], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(wv__ssa_v0, [k0__ssa_v2, q0__ssa_v0], [4, 8], [4, 8], target_memory=pl.Mem.Mat, transpose=False)
                        t__tile_11: pl.Tile[[2, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(n_chunk__tile, wq_c__tile)
                        q_acc__tile_2: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(q_acc__iter_v2, t__tile_11)
                        t__tile_12: pl.Tile[[2, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(n_chunk__tile, wk_c__tile)
                        k_acc__tile_2: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(k_acc__iter_v2, t__tile_12)
                        t__tile_13: pl.Tile[[2, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(n_chunk__tile, wv_c__tile)
                        v_acc__tile_2: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(v_acc__iter_v2, t__tile_13)
                        k_acc__rv_v3, q_acc__rv_v3, v_acc__rv_v3 = pl.yield_(k_acc__tile_2, q_acc__tile_2, v_acc__tile_2)
                    t__tile_14: pl.Tile[[2, 8], pl.BF16, pl.Mem.Vec] = pl.tile.cast(q_acc__rv_v3, target_type=pl.BF16, mode='round')
                    q_proj_tile__tile_1: pl.Tile[[2, 80], pl.BF16, pl.Mem.Vec] = pl.tile.assemble(q_proj_tile__iter_v1, t__tile_14, [0, q0__ssa_v0])
                    t__tile_15: pl.Tile[[2, 8], pl.BF16, pl.Mem.Vec] = pl.tile.cast(k_acc__rv_v3, target_type=pl.BF16, mode='round')
                    k_proj_tile__tile_1: pl.Tile[[2, 80], pl.BF16, pl.Mem.Vec] = pl.tile.assemble(k_proj_tile__iter_v1, t__tile_15, [0, q0__ssa_v0])
                    t__tile_16: pl.Tile[[2, 8], pl.BF16, pl.Mem.Vec] = pl.tile.cast(v_acc__rv_v3, target_type=pl.BF16, mode='round')
                    v_proj_tile__tile_1: pl.Tile[[2, 80], pl.BF16, pl.Mem.Vec] = pl.tile.assemble(v_proj_tile__iter_v1, t__tile_16, [0, q0__ssa_v0])
                    k_proj_tile__rv_v2, q_proj_tile__rv_v2, v_proj_tile__rv_v2 = pl.yield_(k_proj_tile__tile_1, q_proj_tile__tile_1, v_proj_tile__tile_1)
                scores__tile: pl.Tile[[2, 2], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 2], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                scores__tile_1: pl.Tile[[2, 2], pl.FP32, pl.Mem.Vec] = pl.tile.muls(scores__tile, 0.0)
                for kb__idx_v0_3, (btrans_scratch__iter_v5, scores__iter_v2) in pl.range(20, init_values=(btrans_scratch__iter_v3, scores__tile_1)):
                    k0__ssa_v3: pl.Scalar[pl.INDEX] = kb__idx_v0_3 * 4
                    t__tile_17: pl.Tile[[2, 4], pl.BF16, pl.Mem.Vec] = pl.tile.slice(q_proj_tile__rv_v2, [2, 4], [0, k0__ssa_v3])
                    q_c__tile: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.cast(t__tile_17, target_type=pl.FP32, mode='round')
                    t__tile_18: pl.Tile[[2, 4], pl.BF16, pl.Mem.Vec] = pl.tile.slice(k_proj_tile__rv_v2, [2, 4], [0, k0__ssa_v3])
                    k_c__tile: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.cast(t__tile_18, target_type=pl.FP32, mode='round')
                    btrans_scratch__tile: pl.Tensor[[2, 4], pl.FP32] = pl.tile.store(k_c__tile, [0, 0], btrans_scratch__iter_v5)
                    k_c_t__tile: pl.Tile[[4, 2], pl.FP32, pl.Mem.Mat, pl.TileView(slayout=pl.TileLayout.col_major)] = pl.tile.load(btrans_scratch__tile, [0, 0], [4, 2], [4, 2], target_memory=pl.Mem.Mat, transpose=True)
                    t__tile_19: pl.Tile[[2, 2], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(q_c__tile, k_c_t__tile)
                    scores__tile_2: pl.Tile[[2, 2], pl.FP32, pl.Mem.Vec] = pl.tile.add(scores__iter_v2, t__tile_19)
                    btrans_scratch__rv_v6, scores__rv_v3 = pl.yield_(btrans_scratch__tile, scores__tile_2)
                scores__tile_3: pl.Tile[[2, 2], pl.FP32, pl.Mem.Vec] = pl.tile.muls(scores__rv_v3, 0.111803)
                scores_exp__tile: pl.Tile[[2, 2], pl.FP32, pl.Mem.Vec] = pl.tile.exp(scores__tile_3)
                tmp_tile_1: pl.Tile[[2, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                scores_sum__tile: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.row_sum(scores_exp__tile, tmp_tile_1)
                t__tile_20: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.recip(scores_sum__tile)
                attn_w__tile: pl.Tile[[2, 2], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(scores_exp__tile, t__tile_20)
                context_tile__tile: pl.Tile[[2, 80], pl.BF16, pl.Mem.Vec] = pl.tile.create([2, 80], dtype=pl.BF16, target_memory=pl.Mem.Vec)
                for ob__idx_v0_1, (context_tile__iter_v1,) in pl.range(10, init_values=(context_tile__tile,)):
                    o0__ssa_v0: pl.Scalar[pl.INDEX] = ob__idx_v0_1 * 8
                    t__tile_21: pl.Tile[[2, 8], pl.BF16, pl.Mem.Vec] = pl.tile.slice(v_proj_tile__rv_v2, [2, 8], [0, o0__ssa_v0])
                    v_c__tile: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.cast(t__tile_21, target_type=pl.FP32, mode='round')
                    ctx_acc__tile: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    ctx_acc__tile_1: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(ctx_acc__tile, 0.0)
                    t__tile_22: pl.Tile[[2, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(attn_w__tile, v_c__tile)
                    ctx_acc__tile_2: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(ctx_acc__tile_1, t__tile_22)
                    t__tile_23: pl.Tile[[2, 8], pl.BF16, pl.Mem.Vec] = pl.tile.cast(ctx_acc__tile_2, target_type=pl.BF16, mode='round')
                    context_tile__tile_1: pl.Tile[[2, 80], pl.BF16, pl.Mem.Vec] = pl.tile.assemble(context_tile__iter_v1, t__tile_23, [0, o0__ssa_v0])
                    context_tile__rv_v2: pl.Tile[[2, 80], pl.BF16, pl.Mem.Vec] = pl.yield_(context_tile__tile_1)
                resid1_tile__tile: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 80], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                for ob__idx_v0_2, (resid1_tile__iter_v1,) in pl.range(10, init_values=(resid1_tile__tile,)):
                    o0__ssa_v1: pl.Scalar[pl.INDEX] = ob__idx_v0_2 * 8
                    o_acc__tile: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    o_acc__tile_1: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(o_acc__tile, 0.0)
                    for kb__idx_v0_4, (o_acc__iter_v2,) in pl.range(20, init_values=(o_acc__tile_1,)):
                        k0__ssa_v4: pl.Scalar[pl.INDEX] = kb__idx_v0_4 * 4
                        ctx_chunk__tile: pl.Tile[[2, 4], pl.BF16, pl.Mem.Vec] = pl.tile.slice(context_tile__rv_v2, [2, 4], [0, k0__ssa_v4])
                        wo_c__tile: pl.Tile[[4, 8], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(wo__ssa_v0, [k0__ssa_v4, o0__ssa_v1], [4, 8], [4, 8], target_memory=pl.Mem.Mat, transpose=False)
                        t__tile_24: pl.Tile[[2, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(ctx_chunk__tile, wo_c__tile)
                        o_acc__tile_2: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(o_acc__iter_v2, t__tile_24)
                        o_acc__rv_v3: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.yield_(o_acc__tile_2)
                    t__tile_25: pl.Tile[[2, 8], pl.BF16, pl.Mem.Vec] = pl.tile.load(hidden_states__ssa_v0, [0 + b__cr_idx_v0 * 1, p0__ssa_v0, o0__ssa_v1], [1, 2, 8], [1, 2, 8], target_memory=pl.Mem.Vec, transpose=False)
                    t__tile_26: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.cast(t__tile_25, target_type=pl.FP32, mode='round')
                    resid__tile: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_26, [2, 8])
                    t__tile_27: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(o_acc__rv_v3, resid__tile)
                    resid1_tile__tile_1: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(resid1_tile__iter_v1, t__tile_27, [0, o0__ssa_v1])
                    resid1_tile__rv_v2: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.yield_(resid1_tile__tile_1)
                sq_sum2__tile: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.create([2, 1], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                sq_sum2__tile_1: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(sq_sum2__tile, 0.0)
                for kb__idx_v0_5, (sq_sum2__iter_v2,) in pl.range(20, init_values=(sq_sum2__tile_1,)):
                    k0__ssa_v5: pl.Scalar[pl.INDEX] = kb__idx_v0_5 * 4
                    x_chunk__tile_2: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.slice(resid1_tile__rv_v2, [2, 4], [0, k0__ssa_v5])
                    t__tile_28: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.mul(x_chunk__tile_2, x_chunk__tile_2)
                    tmp_tile_2: pl.Tile[[2, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    t__tile_29: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.row_sum(t__tile_28, tmp_tile_2)
                    sq_sum2__tile_2: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.add(sq_sum2__iter_v2, t__tile_29)
                    sq_sum2__rv_v3: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.yield_(sq_sum2__tile_2)
                t__tile_30: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(sq_sum2__rv_v3, 0.0125)
                t__tile_31: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.adds(t__tile_30, 1e-06)
                inv_rms2__tile: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.rsqrt(t__tile_31)
                post_norm_tile__tile: pl.Tile[[2, 80], pl.BF16, pl.Mem.Vec] = pl.tile.create([2, 80], dtype=pl.BF16, target_memory=pl.Mem.Vec)
                for kb__idx_v0_6, (post_norm_tile__iter_v1,) in pl.range(20, init_values=(post_norm_tile__tile,)):
                    k0__ssa_v6: pl.Scalar[pl.INDEX] = kb__idx_v0_6 * 4
                    x_chunk__tile_3: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.slice(resid1_tile__rv_v2, [2, 4], [0, k0__ssa_v6])
                    gamma__tile_1: pl.Tile[[1, 4], pl.FP32, pl.Mem.Vec] = pl.tile.load(post_rms_weight__ssa_v0, [0, k0__ssa_v6], [1, 4], [1, 4], target_memory=pl.Mem.Vec, transpose=False)
                    t__tile_32: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(x_chunk__tile_3, inv_rms2__tile)
                    normed__tile_1: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_32, gamma__tile_1)
                    t__tile_33: pl.Tile[[2, 4], pl.BF16, pl.Mem.Vec] = pl.tile.cast(normed__tile_1, target_type=pl.BF16, mode='round')
                    post_norm_tile__tile_1: pl.Tile[[2, 80], pl.BF16, pl.Mem.Vec] = pl.tile.assemble(post_norm_tile__iter_v1, t__tile_33, [0, k0__ssa_v6])
                    post_norm_tile__rv_v2: pl.Tile[[2, 80], pl.BF16, pl.Mem.Vec] = pl.yield_(post_norm_tile__tile_1)
                down_tile__tile: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 80], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                down_tile__tile_1: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.tile.muls(down_tile__tile, 0.0)
                for mb__idx_v0, (down_tile__iter_v2,) in pl.range(25, init_values=(down_tile__tile_1,)):
                    m0__ssa_v0: pl.Scalar[pl.INDEX] = mb__idx_v0 * 16
                    gate_acc__tile: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    up_acc__tile: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    gate_acc__tile_1: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(gate_acc__tile, 0.0)
                    up_acc__tile_1: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(up_acc__tile, 0.0)
                    for kb__idx_v0_7, (gate_acc__iter_v2, up_acc__iter_v2) in pl.range(20, init_values=(gate_acc__tile_1, up_acc__tile_1)):
                        k0__ssa_v7: pl.Scalar[pl.INDEX] = kb__idx_v0_7 * 4
                        post_chunk__tile: pl.Tile[[2, 4], pl.BF16, pl.Mem.Vec] = pl.tile.slice(post_norm_tile__rv_v2, [2, 4], [0, k0__ssa_v7])
                        wg__tile: pl.Tile[[4, 16], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(w_gate__ssa_v0, [k0__ssa_v7, m0__ssa_v0], [4, 16], [4, 16], target_memory=pl.Mem.Mat, transpose=False)
                        wu__tile: pl.Tile[[4, 16], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(w_up__ssa_v0, [k0__ssa_v7, m0__ssa_v0], [4, 16], [4, 16], target_memory=pl.Mem.Mat, transpose=False)
                        t__tile_34: pl.Tile[[2, 16], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(post_chunk__tile, wg__tile)
                        gate_acc__tile_2: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(gate_acc__iter_v2, t__tile_34)
                        t__tile_35: pl.Tile[[2, 16], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(post_chunk__tile, wu__tile)
                        up_acc__tile_2: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(up_acc__iter_v2, t__tile_35)
                        gate_acc__rv_v3, up_acc__rv_v3 = pl.yield_(gate_acc__tile_2, up_acc__tile_2)
                    t__tile_36: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.neg(gate_acc__rv_v3)
                    t__tile_37: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.exp(t__tile_36)
                    t__tile_38: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(t__tile_37, 1.0)
                    sigmoid_chunk__tile: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.recip(t__tile_38)
                    t__tile_39: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.mul(gate_acc__rv_v3, sigmoid_chunk__tile)
                    t__tile_40: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.mul(t__tile_39, up_acc__rv_v3)
                    mlp_chunk__tile: pl.Tile[[2, 16], pl.BF16, pl.Mem.Vec] = pl.tile.cast(t__tile_40, target_type=pl.BF16, mode='round')
                    for ob__idx_v0_3, (down_tile__iter_v4,) in pl.range(10, init_values=(down_tile__iter_v2,)):
                        o0__ssa_v2: pl.Scalar[pl.INDEX] = ob__idx_v0_3 * 8
                        down_prev__tile: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.slice(down_tile__iter_v4, [2, 8], [0, o0__ssa_v2])
                        wd__tile: pl.Tile[[16, 8], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(w_down__ssa_v0, [m0__ssa_v0, o0__ssa_v2], [16, 8], [16, 8], target_memory=pl.Mem.Mat, transpose=False)
                        t__tile_41: pl.Tile[[2, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(mlp_chunk__tile, wd__tile)
                        down_part__tile: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(down_prev__tile, t__tile_41)
                        down_tile__tile_2: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(down_tile__iter_v4, down_part__tile, [0, o0__ssa_v2])
                        down_tile__rv_v5: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.yield_(down_tile__tile_2)
                    down_tile__rv_v3: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.yield_(down_tile__rv_v5)
                out_tile__tile: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 80], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                for ob__idx_v0_4, (out__iter_v5, out_tile__iter_v1) in pl.range(10, init_values=(out__iter_v3, out_tile__tile)):
                    o0__ssa_v3: pl.Scalar[pl.INDEX] = ob__idx_v0_4 * 8
                    t__tile_42: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.slice(down_tile__rv_v3, [2, 8], [0, o0__ssa_v3])
                    t__tile_43: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.slice(resid1_tile__rv_v2, [2, 8], [0, o0__ssa_v3])
                    out_chunk__tile: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_42, t__tile_43)
                    out_tile__tile_1: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(out_tile__iter_v1, out_chunk__tile, [0, o0__ssa_v3])
                    t__tile_44: pl.Tile[[2, 8], pl.BF16, pl.Mem.Vec] = pl.tile.cast(out_chunk__tile, target_type=pl.BF16, mode='round')
                    t__tile_45: pl.Tile[[1, 2, 8], pl.BF16, pl.Mem.Vec] = pl.tile.reshape(t__tile_44, [1, 2, 8])
                    out__tile: pl.Tensor[[1, 4, 80], pl.BF16] = pl.tile.store(t__tile_45, [0 + b__cr_idx_v0 * 1, p0__ssa_v0, o0__ssa_v3], out__iter_v5, [1, 2, 8])
                    out__rv_v6, out_tile__rv_v2 = pl.yield_(out__tile, out_tile__tile_1)
                t__tile_46: pl.Tile[[2, 80], pl.BF16, pl.Mem.Vec] = pl.tile.load(target_states__ssa_v0, [0 + b__cr_idx_v0 * 1, p0__ssa_v0, 0], [1, 2, 80], [1, 2, 80], target_memory=pl.Mem.Vec, transpose=False)
                t__tile_47: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.tile.cast(t__tile_46, target_type=pl.FP32, mode='round')
                tgt_tile__tile: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_47, [2, 80])
                diff_tile__tile: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.tile.sub(out_tile__rv_v2, tgt_tile__tile)
                sq_tile__tile: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.tile.mul(diff_tile__tile, diff_tile__tile)
                tmp_tile_3: pl.Tile[[2, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                sq_row__tile: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.row_sum(sq_tile__tile, tmp_tile_3)
                loss_prev__tile: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.load(loss_acc__iter_v4, [0, 0], [2, 1], [2, 1], target_memory=pl.Mem.Vec, transpose=False)
                t__tile_48: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.add(loss_prev__tile, sq_row__tile)
                loss_acc__tile: pl.Tensor[[2, 1], pl.FP32] = pl.tile.store(t__tile_48, [0, 0], loss_acc__iter_v4)
                d_out__tile: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.tile.muls(diff_tile__tile, 0.025)
                d_down__ssa_v0: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = d_out__tile
                d_resid1_bwd__ssa_v0: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = d_out__tile
                d_post_norm__tile: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 80], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                d_post_norm__tile_1: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.tile.muls(d_post_norm__tile, 0.0)
                for mb__idx_v0_1, (d_post_norm__iter_v2,) in pl.range(25, init_values=(d_post_norm__tile_1,)):
                    m0__ssa_v1: pl.Scalar[pl.INDEX] = mb__idx_v0_1 * 16
                    gate_r__tile: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    up_r__tile: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    gate_r__tile_1: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(gate_r__tile, 0.0)
                    up_r__tile_1: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(up_r__tile, 0.0)
                    for kb__idx_v0_8, (gate_r__iter_v2, up_r__iter_v2) in pl.range(20, init_values=(gate_r__tile_1, up_r__tile_1)):
                        k0__ssa_v8: pl.Scalar[pl.INDEX] = kb__idx_v0_8 * 4
                        post_c__tile: pl.Tile[[2, 4], pl.BF16, pl.Mem.Vec] = pl.tile.slice(post_norm_tile__rv_v2, [2, 4], [0, k0__ssa_v8])
                        wg_c__tile: pl.Tile[[4, 16], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(w_gate__ssa_v0, [k0__ssa_v8, m0__ssa_v1], [4, 16], [4, 16], target_memory=pl.Mem.Mat, transpose=False)
                        wu_c__tile: pl.Tile[[4, 16], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(w_up__ssa_v0, [k0__ssa_v8, m0__ssa_v1], [4, 16], [4, 16], target_memory=pl.Mem.Mat, transpose=False)
                        t__tile_49: pl.Tile[[2, 16], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(post_c__tile, wg_c__tile)
                        gate_r__tile_2: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(gate_r__iter_v2, t__tile_49)
                        t__tile_50: pl.Tile[[2, 16], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(post_c__tile, wu_c__tile)
                        up_r__tile_2: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(up_r__iter_v2, t__tile_50)
                        gate_r__rv_v3, up_r__rv_v3 = pl.yield_(gate_r__tile_2, up_r__tile_2)
                    t__tile_51: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.neg(gate_r__rv_v3)
                    t__tile_52: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.exp(t__tile_51)
                    t__tile_53: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(t__tile_52, 1.0)
                    sig_r__tile: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.recip(t__tile_53)
                    d_mlp__tile: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    d_mlp__tile_1: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(d_mlp__tile, 0.0)
                    for ob__idx_v0_5, (d_mlp__iter_v2,) in pl.range(10, init_values=(d_mlp__tile_1,)):
                        o0__ssa_v4: pl.Scalar[pl.INDEX] = ob__idx_v0_5 * 8
                        t__tile_54: pl.Tile[[2, 8], pl.FP32, pl.Mem.Vec] = pl.tile.slice(d_down__ssa_v0, [2, 8], [0, o0__ssa_v4])
                        dd_c__tile: pl.Tile[[2, 8], pl.BF16, pl.Mem.Vec] = pl.tile.cast(t__tile_54, target_type=pl.BF16, mode='round')
                        wd_c__tile: pl.Tile[[8, 16], pl.BF16, pl.Mem.Mat, pl.TileView(slayout=pl.TileLayout.col_major)] = pl.tile.load(w_down__ssa_v0, [m0__ssa_v1, o0__ssa_v4], [8, 16], [8, 16], target_memory=pl.Mem.Mat, transpose=True)
                        t__tile_55: pl.Tile[[2, 16], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(dd_c__tile, wd_c__tile)
                        d_mlp__tile_2: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(d_mlp__iter_v2, t__tile_55)
                        d_mlp__rv_v3: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.yield_(d_mlp__tile_2)
                    t__tile_56: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(sig_r__tile, -1.0)
                    one_m_sig__tile: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(t__tile_56, 1.0)
                    t__tile_57: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.mul(gate_r__rv_v3, one_m_sig__tile)
                    t__tile_58: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(t__tile_57, 1.0)
                    silu_deriv__tile: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.mul(sig_r__tile, t__tile_58)
                    t__tile_59: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.mul(d_mlp__rv_v3, up_r__rv_v3)
                    t__tile_60: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.mul(t__tile_59, silu_deriv__tile)
                    d_gate_bf16__tile: pl.Tile[[2, 16], pl.BF16, pl.Mem.Vec] = pl.tile.cast(t__tile_60, target_type=pl.BF16, mode='round')
                    t__tile_61: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.mul(gate_r__rv_v3, sig_r__tile)
                    t__tile_62: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.mul(d_mlp__rv_v3, t__tile_61)
                    d_up_bf16__tile: pl.Tile[[2, 16], pl.BF16, pl.Mem.Vec] = pl.tile.cast(t__tile_62, target_type=pl.BF16, mode='round')
                    for kb__idx_v0_9, (d_post_norm__iter_v4,) in pl.range(20, init_values=(d_post_norm__iter_v2,)):
                        k0__ssa_v9: pl.Scalar[pl.INDEX] = kb__idx_v0_9 * 4
                        dpn_old__tile: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.slice(d_post_norm__iter_v4, [2, 4], [0, k0__ssa_v9])
                        wg_c__tile_1: pl.Tile[[16, 4], pl.BF16, pl.Mem.Mat, pl.TileView(slayout=pl.TileLayout.col_major)] = pl.tile.load(w_gate__ssa_v0, [k0__ssa_v9, m0__ssa_v1], [16, 4], [16, 4], target_memory=pl.Mem.Mat, transpose=True)
                        wu_c__tile_1: pl.Tile[[16, 4], pl.BF16, pl.Mem.Mat, pl.TileView(slayout=pl.TileLayout.col_major)] = pl.tile.load(w_up__ssa_v0, [k0__ssa_v9, m0__ssa_v1], [16, 4], [16, 4], target_memory=pl.Mem.Mat, transpose=True)
                        t__tile_63: pl.Tile[[2, 4], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(d_gate_bf16__tile, wg_c__tile_1)
                        dpn_tmp__tile: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.add(dpn_old__tile, t__tile_63)
                        t__tile_64: pl.Tile[[2, 4], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(d_up_bf16__tile, wu_c__tile_1)
                        dpn_new__tile: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.add(dpn_tmp__tile, t__tile_64)
                        d_post_norm__tile_2: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(d_post_norm__iter_v4, dpn_new__tile, [0, k0__ssa_v9])
                        d_post_norm__rv_v5: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.yield_(d_post_norm__tile_2)
                    d_post_norm__rv_v3: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.yield_(d_post_norm__rv_v5)
                d_resid1__tile: pl.Tile[[2, 80], pl.FP32, pl.Mem.Vec] = pl.tile.add(d_resid1_bwd__ssa_v0, d_post_norm__rv_v3)
                bwd_energy__tile: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.create([2, 1], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                bwd_energy__tile_1: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(bwd_energy__tile, 0.0)
                for kb__idx_v0_10, (bwd_energy__iter_v2,) in pl.range(20, init_values=(bwd_energy__tile_1,)):
                    k0__ssa_v10: pl.Scalar[pl.INDEX] = kb__idx_v0_10 * 4
                    dr_c__tile: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.slice(d_resid1__tile, [2, 4], [0, k0__ssa_v10])
                    t__tile_65: pl.Tile[[2, 4], pl.BF16, pl.Mem.Vec] = pl.tile.slice(q_proj_tile__rv_v2, [2, 4], [0, k0__ssa_v10])
                    q_c__tile_1: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.cast(t__tile_65, target_type=pl.FP32, mode='round')
                    t__tile_66: pl.Tile[[2, 4], pl.BF16, pl.Mem.Vec] = pl.tile.slice(k_proj_tile__rv_v2, [2, 4], [0, k0__ssa_v10])
                    k_c__tile_1: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.cast(t__tile_66, target_type=pl.FP32, mode='round')
                    t__tile_67: pl.Tile[[2, 4], pl.BF16, pl.Mem.Vec] = pl.tile.slice(v_proj_tile__rv_v2, [2, 4], [0, k0__ssa_v10])
                    v_bwd__tile: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.cast(t__tile_67, target_type=pl.FP32, mode='round')
                    t__tile_68: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.add(q_c__tile_1, k_c__tile_1)
                    t__tile_69: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_68, v_bwd__tile)
                    t__tile_70: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.mul(dr_c__tile, t__tile_69)
                    tmp_tile_4: pl.Tile[[2, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    contrib__tile: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.row_sum(t__tile_70, tmp_tile_4)
                    bwd_energy__tile_2: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.add(bwd_energy__iter_v2, contrib__tile)
                    bwd_energy__rv_v3: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.yield_(bwd_energy__tile_2)
                tmp_tile_5: pl.Tile[[2, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([2, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_71: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.row_sum(attn_w__tile, tmp_tile_5)
                bwd_energy__tile_3: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.add(bwd_energy__rv_v3, t__tile_71)
                grad_sink__tile: pl.Tile[[2, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(bwd_energy__tile_3, 0.0)
                btrans_scratch__rv_v4, loss_acc__rv_v5, out__rv_v4 = pl.yield_(btrans_scratch__rv_v6, loss_acc__tile, out__rv_v6)
            btrans_scratch__cr_rv_v1, loss_acc__cr_rv_v2, out__cr_rv_v1 = pl.yield_(btrans_scratch__rv_v4, loss_acc__rv_v5, out__rv_v4)
        return loss_acc__cr_rv_v2, out__cr_rv_v1
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_32b_training_forward_and_backward_layer_incore_3(self, t__tmp_v72: pl.Tensor[[2, 16], pl.BF16], ret0__out: pl.Out[pl.Tensor[[2, 16], pl.BF16]]) -> pl.Tensor[[2, 16], pl.BF16]:
        t__tile: pl.Tile[[2, 16], pl.BF16, pl.Mem.Vec] = pl.tile.load(t__tmp_v72, [0, 0], [2, 16], [2, 16], target_memory=pl.Mem.Vec, transpose=False)
        proxy_mlp__tile: pl.Tile[[2, 16], pl.BF16, pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.BF16, mode='round')
        ret0__store: pl.Tensor[[2, 16], pl.BF16] = pl.tile.store(proxy_mlp__tile, [0, 0], ret0__out)
        return ret0__store
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_32b_training_forward_and_backward_layer_incore_4(self, grad_w_down__ssa_v1: pl.Out[pl.Tensor[[400, 80], pl.FP32]], mom_w_down__ssa_v0: pl.InOut[pl.Tensor[[400, 80], pl.FP32]], muon_scratch__ssa_v0: pl.InOut[pl.Tensor[[16, 16], pl.BF16]], proxy_mlp_t__ssa_v0: pl.Tensor[[2, 16], pl.BF16], proxy_scratch__ssa_v1: pl.InOut[pl.Tensor[[2, 16], pl.BF16]], target_states__ssa_v0: pl.Tensor[[1, 4, 80], pl.BF16]) -> tuple[pl.Tensor[[400, 80], pl.FP32], pl.Tensor[[400, 80], pl.FP32], pl.Tensor[[16, 16], pl.BF16], pl.Tensor[[2, 16], pl.BF16]]:
        for qb__idx_v0, (grad_w_down__iter_v2, mom_w_down__iter_v1, muon_scratch__iter_v1, proxy_scratch__iter_v2) in pl.range(10, init_values=(grad_w_down__ssa_v1, mom_w_down__ssa_v0, muon_scratch__ssa_v0, proxy_scratch__ssa_v1)):
            q0__ssa_v1: pl.Scalar[pl.INDEX] = qb__idx_v0 * 8
            t__tile: pl.Tile[[2, 8], pl.BF16, pl.Mem.Vec] = pl.tile.load(target_states__ssa_v0, [0, 0, q0__ssa_v1], [1, 2, 8], [1, 2, 8], target_memory=pl.Mem.Vec, transpose=False)
            t__tile_1: pl.Tile[[2, 8], pl.BF16, pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.BF16, mode='round')
            proxy_go__tile: pl.Tile[[2, 8], pl.BF16, pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [2, 8])
            proxy_scratch__tile: pl.Tensor[[2, 16], pl.BF16] = pl.tile.store(proxy_go__tile, [0, 0], proxy_scratch__iter_v2)
            proxy_go_t__tile: pl.Tile[[2, 8], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(proxy_scratch__tile, [0, 0], [2, 8], [2, 8], target_memory=pl.Mem.Mat, transpose=False)
            lhs_mat: pl.Tile[[16, 2], pl.BF16, pl.Mem.Mat, pl.TileView(slayout=pl.TileLayout.col_major)] = pl.tile.load(proxy_mlp_t__ssa_v0, [0, 0], [16, 2], [16, 2], target_memory=pl.Mem.Mat, transpose=True)
            grad_down_raw__tile: pl.Tile[[16, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(lhs_mat, proxy_go_t__tile)
            mom_down_prev__tile: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(mom_w_down__iter_v1, [0, q0__ssa_v1], [16, 8], [16, 8], target_memory=pl.Mem.Vec, transpose=False)
            t__tile_2: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(mom_down_prev__tile, 0.95)
            t__tile_3: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(grad_down_raw__tile, 0.05)
            mom_down_new__tile: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_2, t__tile_3)
            muon_down__ssa_v0: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec] = mom_down_new__tile
            t__tile_4: pl.Tile[[16, 8], pl.BF16, pl.Mem.Vec] = pl.tile.cast(muon_down__ssa_v0, target_type=pl.BF16, mode='round')
            muon_scratch__tile: pl.Tensor[[16, 16], pl.BF16] = pl.tile.store(t__tile_4, [0, 0], muon_scratch__iter_v1)
            for ___idx_v0, (muon_down__iter_v1,) in pl.range(2, init_values=(muon_down__ssa_v0,)):
                gram__tile: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                gram__tile_1: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(gram__tile, 0.0)
                for tb__idx_v0, (gram__iter_v2,) in pl.range(8, init_values=(gram__tile_1,)):
                    t0__ssa_v0: pl.Scalar[pl.INDEX] = tb__idx_v0 * 2
                    m_blk__tile: pl.Tile[[2, 8], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(muon_scratch__tile, [t0__ssa_v0, 0], [2, 8], [2, 8], target_memory=pl.Mem.Mat, transpose=False)
                    m_blk_t__tile: pl.Tile[[8, 2], pl.BF16, pl.Mem.Mat] = pl.tile.transpose(m_blk__tile, 0, 1)
                    t__tile_5: pl.Tile[[8, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(m_blk_t__tile, m_blk__tile)
                    gram__tile_2: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(gram__iter_v2, t__tile_5)
                    gram__rv_v3: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.yield_(gram__tile_2)
                t__tile_6: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(muon_down__iter_v1, 1.5)
                t__tile_7: pl.Tile[[16, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(muon_down__iter_v1, gram__rv_v3)
                t__tile_8: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(t__tile_7, -0.5)
                muon_down__tile: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_6, t__tile_8)
                muon_down__rv_v2: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec] = pl.yield_(muon_down__tile)
            t__tile_9: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(muon_down__rv_v2, -0.0002)
            grad_w_down__tile: pl.Tensor[[400, 80], pl.FP32] = pl.tile.store(t__tile_9, [0, q0__ssa_v1], grad_w_down__iter_v2)
            mom_w_down__tile: pl.Tensor[[400, 80], pl.FP32] = pl.tile.store(mom_down_new__tile, [0, q0__ssa_v1], mom_w_down__iter_v1)
            grad_w_down__rv_v3, mom_w_down__rv_v2, muon_scratch__rv_v2, proxy_scratch__rv_v3 = pl.yield_(grad_w_down__tile, mom_w_down__tile, muon_scratch__tile, proxy_scratch__tile)
        return grad_w_down__rv_v3, mom_w_down__rv_v2, muon_scratch__rv_v2, proxy_scratch__rv_v3
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_32b_training_forward_and_backward_layer_incore_5(self, t__tmp_v83: pl.Tensor[[2, 4], pl.BF16], ret0__out: pl.Out[pl.Tensor[[2, 4], pl.BF16]]) -> pl.Tensor[[2, 4], pl.BF16]:
        t__tile: pl.Tile[[2, 4], pl.BF16, pl.Mem.Vec] = pl.tile.load(t__tmp_v83, [0, 0], [2, 4], [2, 4], target_memory=pl.Mem.Vec, transpose=False)
        proxy_ctx__tile: pl.Tile[[2, 4], pl.BF16, pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.BF16, mode='round')
        ret0__store: pl.Tensor[[2, 4], pl.BF16] = pl.tile.store(proxy_ctx__tile, [0, 0], ret0__out)
        return ret0__store
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_32b_training_forward_and_backward_layer_incore_6(self, t__tmp_v84: pl.Tensor[[1, 2, 4], pl.BF16], ret0__out: pl.Out[pl.Tensor[[1, 2, 4], pl.BF16]]) -> pl.Tensor[[1, 2, 4], pl.BF16]:
        t__tile: pl.Tile[[2, 4], pl.BF16, pl.Mem.Vec] = pl.tile.load(t__tmp_v84, [0, 0, 0], [1, 2, 4], [1, 2, 4], target_memory=pl.Mem.Vec, transpose=False)
        t__tile_1: pl.Tile[[2, 4], pl.BF16, pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.BF16, mode='round')
        ret0__store: pl.Tensor[[1, 2, 4], pl.BF16] = pl.tile.store(t__tile_1, [0, 0, 0], ret0__out, [1, 2, 4])
        return ret0__store
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_32b_training_forward_and_backward_layer_incore_7(self, grad_wk__ssa_v1: pl.Out[pl.Tensor[[80, 80], pl.FP32]], grad_wo__ssa_v1: pl.Out[pl.Tensor[[80, 80], pl.FP32]], grad_wq__ssa_v1: pl.Out[pl.Tensor[[80, 80], pl.FP32]], grad_wv__ssa_v1: pl.Out[pl.Tensor[[80, 80], pl.FP32]], mom_wk__ssa_v0: pl.InOut[pl.Tensor[[80, 80], pl.FP32]], mom_wo__ssa_v0: pl.InOut[pl.Tensor[[80, 80], pl.FP32]], mom_wq__ssa_v0: pl.InOut[pl.Tensor[[80, 80], pl.FP32]], mom_wv__ssa_v0: pl.InOut[pl.Tensor[[80, 80], pl.FP32]], muon_scratch__rv_v2: pl.InOut[pl.Tensor[[16, 16], pl.BF16]], proxy_ctx_t__ssa_v0: pl.Tensor[[2, 4], pl.BF16], proxy_n_t__ssa_v0: pl.Tensor[[2, 4], pl.BF16], proxy_scratch__ssa_v6: pl.InOut[pl.Tensor[[2, 16], pl.BF16]], target_states__ssa_v0: pl.Tensor[[1, 4, 80], pl.BF16]) -> tuple[pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[16, 16], pl.BF16], pl.Tensor[[2, 16], pl.BF16]]:
        for qb__idx_v0, (grad_wk__iter_v2, grad_wo__iter_v2, grad_wq__iter_v2, grad_wv__iter_v2, mom_wk__iter_v1, mom_wo__iter_v1, mom_wq__iter_v1, mom_wv__iter_v1, muon_scratch__iter_v4, proxy_scratch__iter_v7) in pl.range(10, init_values=(grad_wk__ssa_v1, grad_wo__ssa_v1, grad_wq__ssa_v1, grad_wv__ssa_v1, mom_wk__ssa_v0, mom_wo__ssa_v0, mom_wq__ssa_v0, mom_wv__ssa_v0, muon_scratch__rv_v2, proxy_scratch__ssa_v6)):
            q0__ssa_v2: pl.Scalar[pl.INDEX] = qb__idx_v0 * 8
            t__tile: pl.Tile[[2, 8], pl.BF16, pl.Mem.Vec] = pl.tile.load(target_states__ssa_v0, [0, 0, q0__ssa_v2], [1, 2, 8], [1, 2, 8], target_memory=pl.Mem.Vec, transpose=False)
            t__tile_1: pl.Tile[[2, 8], pl.BF16, pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.BF16, mode='round')
            proxy_tgt__tile: pl.Tile[[2, 8], pl.BF16, pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [2, 8])
            proxy_scratch__tile: pl.Tensor[[2, 16], pl.BF16] = pl.tile.store(proxy_tgt__tile, [0, 0], proxy_scratch__iter_v7)
            proxy_tgt_t__tile: pl.Tile[[2, 8], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(proxy_scratch__tile, [0, 0], [2, 8], [2, 8], target_memory=pl.Mem.Mat, transpose=False)
            lhs_mat: pl.Tile[[4, 2], pl.BF16, pl.Mem.Mat, pl.TileView(slayout=pl.TileLayout.col_major)] = pl.tile.load(proxy_ctx_t__ssa_v0, [0, 0], [4, 2], [4, 2], target_memory=pl.Mem.Mat, transpose=True)
            grad_wo_raw__tile: pl.Tile[[4, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(lhs_mat, proxy_tgt_t__tile)
            mom_wo_prev__tile: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(mom_wo__iter_v1, [0, q0__ssa_v2], [4, 8], [4, 8], target_memory=pl.Mem.Vec, transpose=False)
            t__tile_2: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(mom_wo_prev__tile, 0.95)
            t__tile_3: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(grad_wo_raw__tile, 0.05)
            mom_wo_new__tile: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_2, t__tile_3)
            muon_wo__ssa_v0: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = mom_wo_new__tile
            t__tile_4: pl.Tile[[4, 8], pl.BF16, pl.Mem.Vec] = pl.tile.cast(muon_wo__ssa_v0, target_type=pl.BF16, mode='round')
            muon_scratch__tile: pl.Tensor[[16, 16], pl.BF16] = pl.tile.store(t__tile_4, [0, 0], muon_scratch__iter_v4)
            for ___idx_v0, (muon_wo__iter_v1,) in pl.range(2, init_values=(muon_wo__ssa_v0,)):
                gram__tile: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                gram__tile_1: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(gram__tile, 0.0)
                for tb__idx_v0, (gram__iter_v7,) in pl.range(2, init_values=(gram__tile_1,)):
                    t0__ssa_v1: pl.Scalar[pl.INDEX] = tb__idx_v0 * 2
                    m_blk__tile: pl.Tile[[2, 8], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(muon_scratch__tile, [t0__ssa_v1, 0], [2, 8], [2, 8], target_memory=pl.Mem.Mat, transpose=False)
                    m_blk_t__tile: pl.Tile[[8, 2], pl.BF16, pl.Mem.Mat] = pl.tile.transpose(m_blk__tile, 0, 1)
                    t__tile_5: pl.Tile[[8, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(m_blk_t__tile, m_blk__tile)
                    gram__tile_2: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(gram__iter_v7, t__tile_5)
                    gram__rv_v8: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.yield_(gram__tile_2)
                t__tile_6: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(muon_wo__iter_v1, 1.5)
                t__tile_7: pl.Tile[[4, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(muon_wo__iter_v1, gram__rv_v8)
                t__tile_8: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(t__tile_7, -0.5)
                muon_wo__tile: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_6, t__tile_8)
                muon_wo__rv_v2: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.yield_(muon_wo__tile)
            t__tile_9: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(muon_wo__rv_v2, -0.0002)
            grad_wo__tile: pl.Tensor[[80, 80], pl.FP32] = pl.tile.store(t__tile_9, [0, q0__ssa_v2], grad_wo__iter_v2)
            mom_wo__tile: pl.Tensor[[80, 80], pl.FP32] = pl.tile.store(mom_wo_new__tile, [0, q0__ssa_v2], mom_wo__iter_v1)
            lhs_mat_1: pl.Tile[[4, 2], pl.BF16, pl.Mem.Mat, pl.TileView(slayout=pl.TileLayout.col_major)] = pl.tile.load(proxy_n_t__ssa_v0, [0, 0], [4, 2], [4, 2], target_memory=pl.Mem.Mat, transpose=True)
            grad_wq_raw__tile: pl.Tile[[4, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(lhs_mat_1, proxy_tgt_t__tile)
            mom_wq_prev__tile: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(mom_wq__iter_v1, [0, q0__ssa_v2], [4, 8], [4, 8], target_memory=pl.Mem.Vec, transpose=False)
            t__tile_10: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(mom_wq_prev__tile, 0.95)
            t__tile_11: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(grad_wq_raw__tile, 0.05)
            mom_wq_new__tile: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_10, t__tile_11)
            muon_wq__ssa_v0: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = mom_wq_new__tile
            t__tile_12: pl.Tile[[4, 8], pl.BF16, pl.Mem.Vec] = pl.tile.cast(muon_wq__ssa_v0, target_type=pl.BF16, mode='round')
            muon_scratch__tile_1: pl.Tensor[[16, 16], pl.BF16] = pl.tile.store(t__tile_12, [0, 0], muon_scratch__tile)
            for ___idx_v0_1, (muon_wq__iter_v1,) in pl.range(2, init_values=(muon_wq__ssa_v0,)):
                gram__tile_3: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                gram__tile_4: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(gram__tile_3, 0.0)
                for tb__idx_v0_1, (gram__iter_v12,) in pl.range(2, init_values=(gram__tile_4,)):
                    t0__ssa_v2: pl.Scalar[pl.INDEX] = tb__idx_v0_1 * 2
                    m_blk__tile_1: pl.Tile[[2, 8], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(muon_scratch__tile_1, [t0__ssa_v2, 0], [2, 8], [2, 8], target_memory=pl.Mem.Mat, transpose=False)
                    m_blk_t__tile_1: pl.Tile[[8, 2], pl.BF16, pl.Mem.Mat] = pl.tile.transpose(m_blk__tile_1, 0, 1)
                    t__tile_13: pl.Tile[[8, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(m_blk_t__tile_1, m_blk__tile_1)
                    gram__tile_5: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(gram__iter_v12, t__tile_13)
                    gram__rv_v13: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.yield_(gram__tile_5)
                t__tile_14: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(muon_wq__iter_v1, 1.5)
                t__tile_15: pl.Tile[[4, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(muon_wq__iter_v1, gram__rv_v13)
                t__tile_16: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(t__tile_15, -0.5)
                muon_wq__tile: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_14, t__tile_16)
                muon_wq__rv_v2: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.yield_(muon_wq__tile)
            t__tile_17: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(muon_wq__rv_v2, -0.0002)
            grad_wq__tile: pl.Tensor[[80, 80], pl.FP32] = pl.tile.store(t__tile_17, [0, q0__ssa_v2], grad_wq__iter_v2)
            mom_wq__tile: pl.Tensor[[80, 80], pl.FP32] = pl.tile.store(mom_wq_new__tile, [0, q0__ssa_v2], mom_wq__iter_v1)
            lhs_mat_2: pl.Tile[[4, 2], pl.BF16, pl.Mem.Mat, pl.TileView(slayout=pl.TileLayout.col_major)] = pl.tile.load(proxy_n_t__ssa_v0, [0, 0], [4, 2], [4, 2], target_memory=pl.Mem.Mat, transpose=True)
            grad_wk_raw__tile: pl.Tile[[4, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(lhs_mat_2, proxy_tgt_t__tile)
            mom_wk_prev__tile: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(mom_wk__iter_v1, [0, q0__ssa_v2], [4, 8], [4, 8], target_memory=pl.Mem.Vec, transpose=False)
            t__tile_18: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(mom_wk_prev__tile, 0.95)
            t__tile_19: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(grad_wk_raw__tile, 0.05)
            mom_wk_new__tile: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_18, t__tile_19)
            muon_wk__ssa_v0: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = mom_wk_new__tile
            t__tile_20: pl.Tile[[4, 8], pl.BF16, pl.Mem.Vec] = pl.tile.cast(muon_wk__ssa_v0, target_type=pl.BF16, mode='round')
            muon_scratch__tile_2: pl.Tensor[[16, 16], pl.BF16] = pl.tile.store(t__tile_20, [0, 0], muon_scratch__tile_1)
            for ___idx_v0_2, (muon_wk__iter_v1,) in pl.range(2, init_values=(muon_wk__ssa_v0,)):
                gram__tile_6: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                gram__tile_7: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(gram__tile_6, 0.0)
                for tb__idx_v0_2, (gram__iter_v17,) in pl.range(2, init_values=(gram__tile_7,)):
                    t0__ssa_v3: pl.Scalar[pl.INDEX] = tb__idx_v0_2 * 2
                    m_blk__tile_2: pl.Tile[[2, 8], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(muon_scratch__tile_2, [t0__ssa_v3, 0], [2, 8], [2, 8], target_memory=pl.Mem.Mat, transpose=False)
                    m_blk_t__tile_2: pl.Tile[[8, 2], pl.BF16, pl.Mem.Mat] = pl.tile.transpose(m_blk__tile_2, 0, 1)
                    t__tile_21: pl.Tile[[8, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(m_blk_t__tile_2, m_blk__tile_2)
                    gram__tile_8: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(gram__iter_v17, t__tile_21)
                    gram__rv_v18: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.yield_(gram__tile_8)
                t__tile_22: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(muon_wk__iter_v1, 1.5)
                t__tile_23: pl.Tile[[4, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(muon_wk__iter_v1, gram__rv_v18)
                t__tile_24: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(t__tile_23, -0.5)
                muon_wk__tile: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_22, t__tile_24)
                muon_wk__rv_v2: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.yield_(muon_wk__tile)
            t__tile_25: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(muon_wk__rv_v2, -0.0002)
            grad_wk__tile: pl.Tensor[[80, 80], pl.FP32] = pl.tile.store(t__tile_25, [0, q0__ssa_v2], grad_wk__iter_v2)
            mom_wk__tile: pl.Tensor[[80, 80], pl.FP32] = pl.tile.store(mom_wk_new__tile, [0, q0__ssa_v2], mom_wk__iter_v1)
            lhs_mat_3: pl.Tile[[4, 2], pl.BF16, pl.Mem.Mat, pl.TileView(slayout=pl.TileLayout.col_major)] = pl.tile.load(proxy_n_t__ssa_v0, [0, 0], [4, 2], [4, 2], target_memory=pl.Mem.Mat, transpose=True)
            grad_wv_raw__tile: pl.Tile[[4, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(lhs_mat_3, proxy_tgt_t__tile)
            mom_wv_prev__tile: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(mom_wv__iter_v1, [0, q0__ssa_v2], [4, 8], [4, 8], target_memory=pl.Mem.Vec, transpose=False)
            t__tile_26: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(mom_wv_prev__tile, 0.95)
            t__tile_27: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(grad_wv_raw__tile, 0.05)
            mom_wv_new__tile: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_26, t__tile_27)
            muon_wv__ssa_v0: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = mom_wv_new__tile
            t__tile_28: pl.Tile[[4, 8], pl.BF16, pl.Mem.Vec] = pl.tile.cast(muon_wv__ssa_v0, target_type=pl.BF16, mode='round')
            muon_scratch__tile_3: pl.Tensor[[16, 16], pl.BF16] = pl.tile.store(t__tile_28, [0, 0], muon_scratch__tile_2)
            for ___idx_v0_3, (muon_wv__iter_v1,) in pl.range(2, init_values=(muon_wv__ssa_v0,)):
                gram__tile_9: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                gram__tile_10: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(gram__tile_9, 0.0)
                for tb__idx_v0_3, (gram__iter_v22,) in pl.range(2, init_values=(gram__tile_10,)):
                    t0__ssa_v4: pl.Scalar[pl.INDEX] = tb__idx_v0_3 * 2
                    m_blk__tile_3: pl.Tile[[2, 8], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(muon_scratch__tile_3, [t0__ssa_v4, 0], [2, 8], [2, 8], target_memory=pl.Mem.Mat, transpose=False)
                    m_blk_t__tile_3: pl.Tile[[8, 2], pl.BF16, pl.Mem.Mat] = pl.tile.transpose(m_blk__tile_3, 0, 1)
                    t__tile_29: pl.Tile[[8, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(m_blk_t__tile_3, m_blk__tile_3)
                    gram__tile_11: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(gram__iter_v22, t__tile_29)
                    gram__rv_v23: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.yield_(gram__tile_11)
                t__tile_30: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(muon_wv__iter_v1, 1.5)
                t__tile_31: pl.Tile[[4, 8], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(muon_wv__iter_v1, gram__rv_v23)
                t__tile_32: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(t__tile_31, -0.5)
                muon_wv__tile: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_30, t__tile_32)
                muon_wv__rv_v2: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.yield_(muon_wv__tile)
            t__tile_33: pl.Tile[[4, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(muon_wv__rv_v2, -0.0002)
            grad_wv__tile: pl.Tensor[[80, 80], pl.FP32] = pl.tile.store(t__tile_33, [0, q0__ssa_v2], grad_wv__iter_v2)
            mom_wv__tile: pl.Tensor[[80, 80], pl.FP32] = pl.tile.store(mom_wv_new__tile, [0, q0__ssa_v2], mom_wv__iter_v1)
            grad_wk__rv_v3, grad_wo__rv_v3, grad_wq__rv_v3, grad_wv__rv_v3, mom_wk__rv_v2, mom_wo__rv_v2, mom_wq__rv_v2, mom_wv__rv_v2, muon_scratch__rv_v5, proxy_scratch__rv_v8 = pl.yield_(grad_wk__tile, grad_wo__tile, grad_wq__tile, grad_wv__tile, mom_wk__tile, mom_wo__tile, mom_wq__tile, mom_wv__tile, muon_scratch__tile_3, proxy_scratch__tile)
        return grad_wk__rv_v3, grad_wo__rv_v3, grad_wq__rv_v3, grad_wv__rv_v3, mom_wk__rv_v2, mom_wo__rv_v2, mom_wq__rv_v2, mom_wv__rv_v2, muon_scratch__rv_v5, proxy_scratch__rv_v8
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_32b_training_forward_and_backward_layer_incore_8(self, t__tmp_v120: pl.Tensor[[1, 2, 4], pl.BF16], ret0__out: pl.Out[pl.Tensor[[1, 2, 4], pl.BF16]]) -> pl.Tensor[[1, 2, 4], pl.BF16]:
        t__tile: pl.Tile[[2, 4], pl.BF16, pl.Mem.Vec] = pl.tile.load(t__tmp_v120, [0, 0, 0], [1, 2, 4], [1, 2, 4], target_memory=pl.Mem.Vec, transpose=False)
        t__tile_1: pl.Tile[[2, 4], pl.BF16, pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.BF16, mode='round')
        ret0__store: pl.Tensor[[1, 2, 4], pl.BF16] = pl.tile.store(t__tile_1, [0, 0, 0], ret0__out, [1, 2, 4])
        return ret0__store
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_32b_training_forward_and_backward_layer_incore_9(self, grad_w_gate__ssa_v1: pl.Out[pl.Tensor[[80, 400], pl.FP32]], grad_w_up__ssa_v1: pl.Out[pl.Tensor[[80, 400], pl.FP32]], mom_w_gate__ssa_v0: pl.InOut[pl.Tensor[[80, 400], pl.FP32]], mom_w_up__ssa_v0: pl.InOut[pl.Tensor[[80, 400], pl.FP32]], muon_scratch__rv_v5: pl.InOut[pl.Tensor[[16, 16], pl.BF16]], proxy_post_t__ssa_v0: pl.Tensor[[2, 4], pl.BF16], proxy_scratch__ssa_v10: pl.InOut[pl.Tensor[[2, 16], pl.BF16]], w_gate__ssa_v0: pl.Tensor[[80, 400], pl.BF16], w_up__ssa_v0: pl.Tensor[[80, 400], pl.BF16]) -> tuple[pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 400], pl.FP32]]:
        for mb__idx_v0, (grad_w_gate__iter_v2, grad_w_up__iter_v2, mom_w_gate__iter_v1, mom_w_up__iter_v1, muon_scratch__iter_v10, proxy_scratch__iter_v11) in pl.range(25, init_values=(grad_w_gate__ssa_v1, grad_w_up__ssa_v1, mom_w_gate__ssa_v0, mom_w_up__ssa_v0, muon_scratch__rv_v5, proxy_scratch__ssa_v10)):
            m0__ssa_v2: pl.Scalar[pl.INDEX] = mb__idx_v0 * 16
            t__tile: pl.Tile[[2, 16], pl.BF16, pl.Mem.Vec] = pl.tile.load(w_gate__ssa_v0, [0, m0__ssa_v2], [2, 16], [2, 16], target_memory=pl.Mem.Vec, transpose=False)
            proxy_gg__tile: pl.Tile[[2, 16], pl.BF16, pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.BF16, mode='round')
            t__tile_1: pl.Tile[[2, 16], pl.BF16, pl.Mem.Vec] = pl.tile.load(w_up__ssa_v0, [0, m0__ssa_v2], [2, 16], [2, 16], target_memory=pl.Mem.Vec, transpose=False)
            proxy_gu__tile: pl.Tile[[2, 16], pl.BF16, pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.BF16, mode='round')
            proxy_scratch__tile: pl.Tensor[[2, 16], pl.BF16] = pl.tile.store(proxy_gg__tile, [0, 0], proxy_scratch__iter_v11)
            proxy_gg_t__tile: pl.Tile[[2, 16], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(proxy_scratch__tile, [0, 0], [2, 16], [2, 16], target_memory=pl.Mem.Mat, transpose=False)
            proxy_scratch__tile_1: pl.Tensor[[2, 16], pl.BF16] = pl.tile.store(proxy_gu__tile, [0, 0], proxy_scratch__tile)
            proxy_gu_t__tile: pl.Tile[[2, 16], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(proxy_scratch__tile_1, [0, 0], [2, 16], [2, 16], target_memory=pl.Mem.Mat, transpose=False)
            lhs_mat: pl.Tile[[4, 2], pl.BF16, pl.Mem.Mat, pl.TileView(slayout=pl.TileLayout.col_major)] = pl.tile.load(proxy_post_t__ssa_v0, [0, 0], [4, 2], [4, 2], target_memory=pl.Mem.Mat, transpose=True)
            grad_wg_raw__tile: pl.Tile[[4, 16], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(lhs_mat, proxy_gg_t__tile)
            lhs_mat_1: pl.Tile[[4, 2], pl.BF16, pl.Mem.Mat, pl.TileView(slayout=pl.TileLayout.col_major)] = pl.tile.load(proxy_post_t__ssa_v0, [0, 0], [4, 2], [4, 2], target_memory=pl.Mem.Mat, transpose=True)
            grad_wu_raw__tile: pl.Tile[[4, 16], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(lhs_mat_1, proxy_gu_t__tile)
            mom_wg_prev__tile: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(mom_w_gate__iter_v1, [0, m0__ssa_v2], [4, 16], [4, 16], target_memory=pl.Mem.Vec, transpose=False)
            mom_wu_prev__tile: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(mom_w_up__iter_v1, [0, m0__ssa_v2], [4, 16], [4, 16], target_memory=pl.Mem.Vec, transpose=False)
            t__tile_2: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(mom_wg_prev__tile, 0.95)
            t__tile_3: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(grad_wg_raw__tile, 0.05)
            mom_wg_new__tile: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_2, t__tile_3)
            t__tile_4: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(mom_wu_prev__tile, 0.95)
            t__tile_5: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(grad_wu_raw__tile, 0.05)
            mom_wu_new__tile: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_4, t__tile_5)
            muon_wg__ssa_v0: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = mom_wg_new__tile
            t__tile_6: pl.Tile[[4, 16], pl.BF16, pl.Mem.Vec] = pl.tile.cast(muon_wg__ssa_v0, target_type=pl.BF16, mode='round')
            muon_scratch__tile: pl.Tensor[[16, 16], pl.BF16] = pl.tile.store(t__tile_6, [0, 0], muon_scratch__iter_v10)
            for ___idx_v0, (muon_wg__iter_v1,) in pl.range(2, init_values=(muon_wg__ssa_v0,)):
                ns_acc_g__tile: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.tile.create([4, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                ns_acc_g__tile_1: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(ns_acc_g__tile, 0.0)
                muon_wg_bf__tile: pl.Tile[[4, 16], pl.BF16, pl.Mem.Vec] = pl.tile.cast(muon_wg__iter_v1, target_type=pl.BF16, mode='round')
                for tb__idx_v0, (ns_acc_g__iter_v2,) in pl.range(2, init_values=(ns_acc_g__tile_1,)):
                    t0__ssa_v5: pl.Scalar[pl.INDEX] = tb__idx_v0 * 2
                    m_blk_g__tile: pl.Tile[[2, 16], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(muon_scratch__tile, [t0__ssa_v5, 0], [2, 16], [2, 16], target_memory=pl.Mem.Mat, transpose=False)
                    m_blk_gt__tile: pl.Tile[[16, 2], pl.BF16, pl.Mem.Mat] = pl.tile.transpose(m_blk_g__tile, 0, 1)
                    tmp_g__tile: pl.Tile[[4, 2], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(muon_wg_bf__tile, m_blk_gt__tile)
                    tmp_g_bf__tile: pl.Tile[[4, 2], pl.BF16, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.cast(tmp_g__tile, target_type=pl.BF16, mode='round')
                    t__tile_7: pl.Tile[[4, 16], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(tmp_g_bf__tile, m_blk_g__tile)
                    ns_acc_g__tile_2: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(ns_acc_g__iter_v2, t__tile_7)
                    ns_acc_g__rv_v3: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.yield_(ns_acc_g__tile_2)
                t__tile_8: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(muon_wg__iter_v1, 1.5)
                t__tile_9: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(ns_acc_g__rv_v3, -0.5)
                muon_wg__tile: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_8, t__tile_9)
                muon_wg__rv_v2: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.yield_(muon_wg__tile)
            muon_wu__ssa_v0: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = mom_wu_new__tile
            t__tile_10: pl.Tile[[4, 16], pl.BF16, pl.Mem.Vec] = pl.tile.cast(muon_wu__ssa_v0, target_type=pl.BF16, mode='round')
            muon_scratch__tile_1: pl.Tensor[[16, 16], pl.BF16] = pl.tile.store(t__tile_10, [0, 0], muon_scratch__tile)
            for ___idx_v0_1, (muon_wu__iter_v1,) in pl.range(2, init_values=(muon_wu__ssa_v0,)):
                ns_acc_u__tile: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.tile.create([4, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                ns_acc_u__tile_1: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(ns_acc_u__tile, 0.0)
                muon_wu_bf__tile: pl.Tile[[4, 16], pl.BF16, pl.Mem.Vec] = pl.tile.cast(muon_wu__iter_v1, target_type=pl.BF16, mode='round')
                for tb__idx_v0_1, (ns_acc_u__iter_v2,) in pl.range(2, init_values=(ns_acc_u__tile_1,)):
                    t0__ssa_v6: pl.Scalar[pl.INDEX] = tb__idx_v0_1 * 2
                    m_blk_u__tile: pl.Tile[[2, 16], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(muon_scratch__tile_1, [t0__ssa_v6, 0], [2, 16], [2, 16], target_memory=pl.Mem.Mat, transpose=False)
                    m_blk_ut__tile: pl.Tile[[16, 2], pl.BF16, pl.Mem.Mat] = pl.tile.transpose(m_blk_u__tile, 0, 1)
                    tmp_u__tile: pl.Tile[[4, 2], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(muon_wu_bf__tile, m_blk_ut__tile)
                    tmp_u_bf__tile: pl.Tile[[4, 2], pl.BF16, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.cast(tmp_u__tile, target_type=pl.BF16, mode='round')
                    t__tile_11: pl.Tile[[4, 16], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(tmp_u_bf__tile, m_blk_u__tile)
                    ns_acc_u__tile_2: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(ns_acc_u__iter_v2, t__tile_11)
                    ns_acc_u__rv_v3: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.yield_(ns_acc_u__tile_2)
                t__tile_12: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(muon_wu__iter_v1, 1.5)
                t__tile_13: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(ns_acc_u__rv_v3, -0.5)
                muon_wu__tile: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_12, t__tile_13)
                muon_wu__rv_v2: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.yield_(muon_wu__tile)
            t__tile_14: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(muon_wg__rv_v2, -0.0002)
            grad_w_gate__tile: pl.Tensor[[80, 400], pl.FP32] = pl.tile.store(t__tile_14, [0, m0__ssa_v2], grad_w_gate__iter_v2)
            t__tile_15: pl.Tile[[4, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(muon_wu__rv_v2, -0.0002)
            grad_w_up__tile: pl.Tensor[[80, 400], pl.FP32] = pl.tile.store(t__tile_15, [0, m0__ssa_v2], grad_w_up__iter_v2)
            mom_w_gate__tile: pl.Tensor[[80, 400], pl.FP32] = pl.tile.store(mom_wg_new__tile, [0, m0__ssa_v2], mom_w_gate__iter_v1)
            mom_w_up__tile: pl.Tensor[[80, 400], pl.FP32] = pl.tile.store(mom_wu_new__tile, [0, m0__ssa_v2], mom_w_up__iter_v1)
            grad_w_gate__rv_v3, grad_w_up__rv_v3, mom_w_gate__rv_v2, mom_w_up__rv_v2, muon_scratch__rv_v11, proxy_scratch__rv_v12 = pl.yield_(grad_w_gate__tile, grad_w_up__tile, mom_w_gate__tile, mom_w_up__tile, muon_scratch__tile_1, proxy_scratch__tile_1)
        return grad_w_gate__rv_v3, grad_w_up__rv_v3, mom_w_gate__rv_v2, mom_w_up__rv_v2
    @pl.function(type=pl.FunctionType.Orchestration)
    def qwen3_32b_training_forward_and_backward_layer(self, hidden_states__ssa_v0: pl.Tensor[[1, 4, 80], pl.BF16], target_states__ssa_v0: pl.Tensor[[1, 4, 80], pl.BF16], input_rms_weight__ssa_v0: pl.Tensor[[1, 80], pl.FP32], post_rms_weight__ssa_v0: pl.Tensor[[1, 80], pl.FP32], wq__ssa_v0: pl.Tensor[[80, 80], pl.BF16], wk__ssa_v0: pl.Tensor[[80, 80], pl.BF16], wv__ssa_v0: pl.Tensor[[80, 80], pl.BF16], wo__ssa_v0: pl.Tensor[[80, 80], pl.BF16], w_gate__ssa_v0: pl.Tensor[[80, 400], pl.BF16], w_up__ssa_v0: pl.Tensor[[80, 400], pl.BF16], w_down__ssa_v0: pl.Tensor[[400, 80], pl.BF16], mom_wq__ssa_v0: pl.Tensor[[80, 80], pl.FP32], mom_wk__ssa_v0: pl.Tensor[[80, 80], pl.FP32], mom_wv__ssa_v0: pl.Tensor[[80, 80], pl.FP32], mom_wo__ssa_v0: pl.Tensor[[80, 80], pl.FP32], mom_w_gate__ssa_v0: pl.Tensor[[80, 400], pl.FP32], mom_w_up__ssa_v0: pl.Tensor[[80, 400], pl.FP32], mom_w_down__ssa_v0: pl.Tensor[[400, 80], pl.FP32], grad_wq__ssa_v0: pl.Tensor[[80, 80], pl.FP32], grad_wk__ssa_v0: pl.Tensor[[80, 80], pl.FP32], grad_wv__ssa_v0: pl.Tensor[[80, 80], pl.FP32], grad_wo__ssa_v0: pl.Tensor[[80, 80], pl.FP32], grad_w_gate__ssa_v0: pl.Tensor[[80, 400], pl.FP32], grad_w_up__ssa_v0: pl.Tensor[[80, 400], pl.FP32], grad_w_down__ssa_v0: pl.Tensor[[400, 80], pl.FP32], out__ssa_v0: pl.Tensor[[1, 4, 80], pl.BF16], loss_out__ssa_v0: pl.Tensor[[1], pl.FP32]) -> tuple[pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[400, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[400, 80], pl.FP32], pl.Tensor[[1, 4, 80], pl.BF16], pl.Tensor[[1], pl.FP32]]:
        muon_scratch__ssa_v0: pl.Tensor[[16, 16], pl.BF16] = pl.tensor.create([16, 16], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        proxy_scratch__ssa_v0: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.create([2, 16], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        btrans_scratch__ssa_v0: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.create([2, 4], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        ret0__out: pl.Tensor[[400, 80], pl.FP32] = pl.tensor.create([400, 80], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        ret1__out: pl.Tensor[[80, 400], pl.FP32] = pl.tensor.create([80, 400], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        ret2__out: pl.Tensor[[80, 400], pl.FP32] = pl.tensor.create([80, 400], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        ret3__out: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.create([80, 80], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        ret4__out: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.create([80, 80], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        ret5__out: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.create([80, 80], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        ret6__out: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.create([80, 80], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        ret__tmp_v0: pl.Tuple[pl.Tensor[[400, 80], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32]] = self.qwen3_32b_training_forward_and_backward_layer_incore_0(grad_w_down__ssa_v0, grad_w_gate__ssa_v0, grad_w_up__ssa_v0, grad_wk__ssa_v0, grad_wo__ssa_v0, grad_wq__ssa_v0, grad_wv__ssa_v0, ret0__out, ret1__out, ret2__out, ret3__out, ret4__out, ret5__out, ret6__out)
        grad_w_down__ssa_v1: pl.Tensor[[400, 80], pl.FP32] = ret__tmp_v0[0]
        grad_w_gate__ssa_v1: pl.Tensor[[80, 400], pl.FP32] = ret__tmp_v0[1]
        grad_w_up__ssa_v1: pl.Tensor[[80, 400], pl.FP32] = ret__tmp_v0[2]
        grad_wk__ssa_v1: pl.Tensor[[80, 80], pl.FP32] = ret__tmp_v0[3]
        grad_wo__ssa_v1: pl.Tensor[[80, 80], pl.FP32] = ret__tmp_v0[4]
        grad_wq__ssa_v1: pl.Tensor[[80, 80], pl.FP32] = ret__tmp_v0[5]
        grad_wv__ssa_v1: pl.Tensor[[80, 80], pl.FP32] = ret__tmp_v0[6]
        loss_acc__ssa_v0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.create([2, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        ret0__out_1: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.create([2, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        loss_acc__ssa_v1: pl.Tensor[[2, 1], pl.FP32] = self.qwen3_32b_training_forward_and_backward_layer_incore_1(loss_acc__ssa_v0, ret0__out_1)
        tok_blocks__ssa_v0: pl.Scalar[pl.INDEX] = 2
        ret__tmp_v0_1: pl.Tuple[pl.Tensor[[2, 1], pl.FP32], pl.Tensor[[1, 4, 80], pl.BF16]] = self.qwen3_32b_training_forward_and_backward_layer_incore_2(btrans_scratch__ssa_v0, hidden_states__ssa_v0, input_rms_weight__ssa_v0, loss_acc__ssa_v1, out__ssa_v0, post_rms_weight__ssa_v0, target_states__ssa_v0, tok_blocks__ssa_v0, w_down__ssa_v0, w_gate__ssa_v0, w_up__ssa_v0, wk__ssa_v0, wo__ssa_v0, wq__ssa_v0, wv__ssa_v0)
        loss_acc__cr_rv_v2: pl.Tensor[[2, 1], pl.FP32] = ret__tmp_v0_1[0]
        out__cr_rv_v1: pl.Tensor[[1, 4, 80], pl.BF16] = ret__tmp_v0_1[1]
        t__tmp_v72: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.slice(w_up__ssa_v0, [2, 16], [0, 0])
        ret0__out_2: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.create([2, 16], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        proxy_mlp__ssa_v0: pl.Tensor[[2, 16], pl.BF16] = self.qwen3_32b_training_forward_and_backward_layer_incore_3(t__tmp_v72, ret0__out_2)
        proxy_scratch__ssa_v1: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.assemble(proxy_scratch__ssa_v0, proxy_mlp__ssa_v0, [0, 0])
        proxy_mlp_t__ssa_v0: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.slice(proxy_scratch__ssa_v1, [2, 16], [0, 0])
        ret__tmp_v0_2: pl.Tuple[pl.Tensor[[400, 80], pl.FP32], pl.Tensor[[400, 80], pl.FP32], pl.Tensor[[16, 16], pl.BF16], pl.Tensor[[2, 16], pl.BF16]] = self.qwen3_32b_training_forward_and_backward_layer_incore_4(grad_w_down__ssa_v1, mom_w_down__ssa_v0, muon_scratch__ssa_v0, proxy_mlp_t__ssa_v0, proxy_scratch__ssa_v1, target_states__ssa_v0)
        grad_w_down__rv_v3: pl.Tensor[[400, 80], pl.FP32] = ret__tmp_v0_2[0]
        mom_w_down__rv_v2: pl.Tensor[[400, 80], pl.FP32] = ret__tmp_v0_2[1]
        muon_scratch__rv_v2: pl.Tensor[[16, 16], pl.BF16] = ret__tmp_v0_2[2]
        proxy_scratch__rv_v3: pl.Tensor[[2, 16], pl.BF16] = ret__tmp_v0_2[3]
        t__tmp_v83: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.slice(wq__ssa_v0, [2, 4], [0, 0])
        ret0__out_3: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.create([2, 4], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        proxy_ctx__ssa_v0: pl.Tensor[[2, 4], pl.BF16] = self.qwen3_32b_training_forward_and_backward_layer_incore_5(t__tmp_v83, ret0__out_3)
        t__tmp_v84: pl.Tensor[[1, 2, 4], pl.BF16] = pl.tensor.slice(hidden_states__ssa_v0, [1, 2, 4], [0, 0, 0])
        ret0__out_4: pl.Tensor[[1, 2, 4], pl.BF16] = pl.tensor.create([1, 2, 4], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        t__tmp_v85: pl.Tensor[[1, 2, 4], pl.BF16] = self.qwen3_32b_training_forward_and_backward_layer_incore_6(t__tmp_v84, ret0__out_4)
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
        ret0__out_5: pl.Tensor[[1, 2, 4], pl.BF16] = pl.tensor.create([1, 2, 4], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        t__tmp_v121: pl.Tensor[[1, 2, 4], pl.BF16] = self.qwen3_32b_training_forward_and_backward_layer_incore_8(t__tmp_v120, ret0__out_5)
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