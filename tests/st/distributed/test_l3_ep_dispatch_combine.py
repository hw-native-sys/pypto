# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank EP dispatch + local_expert + combine — 1:1 PyPTO
port of ``runtime/examples/workers/l3/ep_dispatch_combine``.

Runs at real DeepSeek-V4 FLASH MoE scale (pypto-lib/models/deepseek/v4):
T=128, TOPK=6, D=4096, L=16 local experts per rank, R=192 receive cap.

This is a structural port of the C++ runtime example. The three AIV kernels
(``dispatch`` / ``local_expert`` / ``combine``) become three
:class:`pl.FunctionType.InCore` kernels chained from ``chip_orch``, with the
same window layout, the same routing protocol, the same per-src signal cells
for every barrier, the same dynamic-loop iteration style, and the same data
direction at every cross-rank op.

* **dispatch**: histogram → publish ``send_counts`` via TNOTIFY(AtomicAdd) +
  count_done barrier (per-src signal cells) → prefix_sum → ``payload_push``
  3-channel push (x BF16 / w FP32 / idx INT32) into peer's
  ``recv_x``/``recv_w``/``recv_idx`` keyed by ``(local_expert, slot)`` →
  data_done barrier → ``stage_out`` window → host-backed
  ``recv_x_out`` / ``recv_w_out`` / ``recv_idx_out``.
* **local_expert**: ``recv_y[e, s, :] = cast_bf16(cast_fp32(recv_x_out) *
  recv_w_out[..., 0])`` with the BF16 round-trip preserved; reads the staged
  host outputs (not the window), mirroring the runtime kernel's argument
  list.
* **combine**: TPUT-style push of ``recv_y[idx_lin, :]`` to peer's
  ``routed_y_buf[r, :]`` where ``r = t * TOPK + k`` from ``recv_idx_out``,
  then combine_done barrier (per-src signal cells), then FP32 reduce_sum
  along TOPK into ``routed_y``.

**Iteration style — 1:1 with runtime.** Every loop uses ``pl.range`` (runtime
``for (int i = 0; i < N; ++i)``) instead of ``pl.unroll``. The runtime kernel
does no compile-time unrolling — neither do we. The natural ``(t, k)``
traversal in payload_push is equivalent to the runtime's sorted route table:
``cursor`` is per-bucket so within-bucket ``(t, k)`` lex order (stable in
either scheme) is all that determines slot assignment, and we skip the
explicit insertion sort.

**Per-src signal cells — 1:1 with runtime.** Every barrier signal
(``count_done`` / ``data_done`` / ``combine_done``) is sized ``[N_RANKS, 1]``;
each rank notifies peer's ``[my_rank, 0]`` cell and waits on its local
``[src, 0]`` for every ``src != my_rank``. Matches ``count_done_sig[N]`` /
``data_done_sig[N]`` / ``combine_done_sig[N]`` in the C++ kernel.

**Stage-out — 1:1 with runtime.** The dispatch kernel emits four host-backed
outputs: ``recv_x_out [L*R, D] BF16``, ``recv_w_out [L, R] FP32``,
``recv_idx_out [L, R] INT32``, and ``recv_count_out [L, 1] INT32``. Per-row
1xD tile copies for x; scalar GM reads of column 0 for w / idx (the wide
window was filled as ``[value, 0, …, 0]`` so column 0 is the real payload).
Downstream kernels read from the staged outputs, not the window.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

# Real DeepSeek-V4 FLASH MoE shapes (pypto-lib/models/deepseek/v4) — must
# mirror the runtime example's constants. ``N_RANKS`` is supplied per-test via
# ``_build_ep_dispatch_combine_program(n_ranks)``; everything below is
# rank-count-independent. T = DECODE_BATCH*DECODE_SEQ, TOPK = experts/tok,
# D = hidden_size, L = N_LOCAL experts/rank, R = RECV_MAX (single-expert
# receive cap).
T = 128
TOPK = 6
D = 4096
L = 16  # N_LOCAL_EXPERTS per rank
R = 192  # RECV_MAX (single-expert receive upper bound)
W_PAD = 8  # weight tile width — minimum vector tile (1x8 FP32 = 32 B)
IDX_PAD = 8  # idx tile width   — minimum vector tile (1x8 INT32 = 32 B)
N_ROUTES = T * TOPK  # 768


def _build_ep_dispatch_combine_program(n_ranks: int):
    """Build the n-rank ep_dispatch_combine program at call time.

    Deferred construction matches other L3 tests — keeps the module importable
    even if the embedded body trips the parser at collection time.
    """
    N_RANKS = n_ranks  # noqa: N806 — closed over by IR shape annotations below

    @pl.program
    class EpDispatchCombine:
        # ----------------------------------------------------------------
        # dispatch — 1:1 of runtime/dispatch.cpp.
        #
        # Reads:   indices, x_norm, w_padded, idx_padded (host-backed inputs)
        # Writes:  recv_x_out / recv_w_out / recv_idx_out / recv_count_out
        #          (host-backed staged outputs)
        #          pub_counts, recv_x, recv_w, recv_idx (window slots)
        # Barriers: count_done (publish→prefix_sum), data_done (push→stage_out)
        # ----------------------------------------------------------------
        @pl.function(type=pl.FunctionType.InCore)
        def dispatch_step(  # noqa: PLR0913, PLR0912, PLR0915
            self,
            indices: pl.Tensor[[T, TOPK], pl.INT32],
            x_norm: pl.Tensor[[T, D], pl.BF16],
            w_padded: pl.Tensor[[N_ROUTES, W_PAD], pl.FP32],
            idx_padded: pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
            recv_x_out: pl.Out[pl.Tensor[[L * R, D], pl.BF16]],
            recv_w_out: pl.Out[pl.Tensor[[L, R], pl.FP32]],
            recv_idx_out: pl.Out[pl.Tensor[[L, R], pl.INT32]],
            recv_count_out: pl.Out[pl.Tensor[[L, 1], pl.INT32]],
            pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, L], pl.INT32],
            count_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            recv_x: pld.DistributedTensor[[L * R, D], pl.BF16],
            recv_w: pld.DistributedTensor[[L * R, W_PAD], pl.FP32],
            recv_idx: pld.DistributedTensor[[L * R, IDX_PAD], pl.INT32],
            data_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> tuple[
            pl.Tensor[[L * R, D], pl.BF16],
            pl.Tensor[[L, R], pl.FP32],
            pl.Tensor[[L, R], pl.INT32],
            pl.Tensor[[L, 1], pl.INT32],
        ]:
            # ---------- histogram: scalar histogram on indices ----------
            send_counts = pl.array.create(N_RANKS * L, pl.INT32)
            for d in pl.range(N_RANKS):
                for e in pl.range(L):
                    send_counts[d * L + e] = 0

            for t in pl.range(T):
                for k in pl.range(TOPK):
                    eid = pl.read(indices, [t, k])
                    d = eid // L
                    e = eid - d * L
                    cur = send_counts[d * L + e]
                    send_counts[d * L + e] = cur + 1

            # ---------- publish: TNOTIFY(AtomicAdd) send_counts to peers ----
            # Each rank publishes its full [N_RANKS, L] send_counts row to
            # every peer's pub_counts[my_rank][:][:] slice. Self-rank is
            # included so pub_counts[my_rank][:][:] gets populated locally.
            for peer in pl.range(N_RANKS):
                for d in pl.range(N_RANKS):
                    for e in pl.range(L):
                        v = send_counts[d * L + e]
                        if v != 0:
                            pld.system.notify(
                                target=pub_counts,
                                peer=peer,
                                offsets=[my_rank * N_RANKS + d, e],
                                value=v,
                                op=pld.NotifyOp.AtomicAdd,
                            )

            # ---------- count_done barrier — per-src signal cells ----------
            # Matches runtime/dispatch.cpp's count_done_sig[N]: notify peer's
            # [my_rank, 0]; wait on local [src, 0] for every src != my_rank.
            for peer in pl.range(N_RANKS):
                if peer != my_rank:
                    pld.system.notify(
                        target=count_done,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(N_RANKS):
                if src != my_rank:
                    pld.system.wait(
                        signal=count_done,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            # ---------- prefix_sum: my_slot_at_dst + recv_count ----------
            # The SSA pass auto-detects `acc` as a loop-carried variable
            # (assigned in body, exists before the loop) and inserts the
            # IterArg/Yield machinery — no manual workaround needed.
            my_slot_at_dst = pl.array.create(N_RANKS * L, pl.INT32)
            for d in pl.range(N_RANKS):
                for e in pl.range(L):
                    acc = pl.const(0, pl.INT32)
                    for s in pl.range(N_RANKS):
                        if s < my_rank:
                            acc = acc + pl.read(pub_counts, [s * N_RANKS + d, e])
                    my_slot_at_dst[d * L + e] = acc

            for e in pl.range(L):
                acc = pl.const(0, pl.INT32)
                for s in pl.range(N_RANKS):
                    acc = acc + pl.read(pub_counts, [s * N_RANKS + my_rank, e])
                pl.write(recv_count_out, [e, 0], acc)

            # ---------- payload_push: natural (t, k) iteration ----------
            # Equivalent to runtime's sorted route-table scan: cursor is
            # per-bucket so within-bucket (t, k) lex order (preserved by
            # either scheme) is all that determines slot assignment.
            cursor = pl.array.create(N_RANKS * L, pl.INT32)
            for d in pl.range(N_RANKS):
                for e in pl.range(L):
                    cursor[d * L + e] = 0

            for t in pl.range(T):
                for k in pl.range(TOPK):
                    eid = pl.read(indices, [t, k])
                    dst = eid // L
                    loc_e = eid - dst * L
                    bucket = dst * L + loc_e
                    cur_val = cursor[bucket]
                    slot_off = my_slot_at_dst[bucket]
                    slot = slot_off + cur_val
                    row = loc_e * R + slot
                    cursor[bucket] = cur_val + 1
                    r_route = t * TOPK + k

                    # Channel 1: x BF16 [1, D] — HCCL TPUT (load + store fused)
                    pld.tensor.put(
                        dst=recv_x,
                        peer=dst,
                        src=x_norm,
                        dst_offsets=[row, 0],
                        src_offsets=[t, 0],
                        shape=[1, D],
                    )

                    # Channel 2: w FP32 [1, W_PAD] — HCCL TPUT
                    pld.tensor.put(
                        dst=recv_w,
                        peer=dst,
                        src=w_padded,
                        dst_offsets=[row, 0],
                        src_offsets=[r_route, 0],
                        shape=[1, W_PAD],
                    )

                    # Channel 3: idx INT32 [1, IDX_PAD] — HCCL TPUT
                    pld.tensor.put(
                        dst=recv_idx,
                        peer=dst,
                        src=idx_padded,
                        dst_offsets=[row, 0],
                        src_offsets=[r_route, 0],
                        shape=[1, IDX_PAD],
                    )

            # ---------- data_done barrier — per-src signal cells ----------
            for peer in pl.range(N_RANKS):
                if peer != my_rank:
                    pld.system.notify(
                        target=data_done,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(N_RANKS):
                if src != my_rank:
                    pld.system.wait(
                        signal=data_done,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            # ---------- stage_out: window → host-backed outputs ----------
            # recv_x_out: per-row 1xD tile copy; mirrors dispatch.cpp's per-row
            # TLOAD/TSTORE loop.
            for e in pl.range(L):
                for slot in pl.range(R):
                    row = e * R + slot
                    x_tile = pl.load(recv_x, [row, 0], [1, D])
                    pl.store(x_tile, [row, 0], recv_x_out)

            # recv_w_out: per-expert TLOAD [R, W_PAD] + row_sum (compacts
            # along W_PAD axis; sum recovers slot [0] since [1, W_PAD) is
            # zero by design) + reshape [R, 1] → [1, R] + TSTORE. Mirrors
            # the runtime's TROWSUM stage_out for the weight channel.
            for e in pl.range(L):
                w_wide: pl.Tile[[R, W_PAD], pl.FP32] = pl.load(recv_w, [e * R, 0], [R, W_PAD])
                tmp: pl.Tile[[R, 1], pl.FP32] = pl.tile.create(
                    [R, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                w_sum: pl.Tile[[R, 1], pl.FP32] = pl.tile.row_sum(w_wide, tmp)
                w_row: pl.Tile[[1, R], pl.FP32] = pl.tile.reshape(w_sum, [1, R])
                pl.store(w_row, [e, 0], recv_w_out)

            # recv_idx_out: scalar copy of column 0 (matches dispatch.cpp's
            # fallback path, which avoids INT32 TROWSUM hangs on a2a3).
            for e in pl.range(L):
                for slot in pl.range(R):
                    r_val = pl.read(recv_idx, [e * R + slot, 0])
                    pl.write(recv_idx_out, [e, slot], r_val)

            return recv_x_out, recv_w_out, recv_idx_out, recv_count_out

        # ----------------------------------------------------------------
        # local_expert — 1:1 of runtime/local_expert.cpp.
        #
        # recv_y[e, slot, :] = cast_bf16(cast_fp32(recv_x_out[e, slot, :]) *
        #                                recv_w_out[e, slot])
        #
        # Reads the staged host-backed outputs. Pure local — no cross-rank ops.
        # ----------------------------------------------------------------
        @pl.function(type=pl.FunctionType.InCore)
        def local_expert_step(
            self,
            recv_x_out: pl.Tensor[[L * R, D], pl.BF16],
            recv_w_out: pl.Tensor[[L, R], pl.FP32],
            recv_count: pl.Tensor[[L, 1], pl.INT32],
            recv_y: pl.Out[pl.Tensor[[L * R, D], pl.BF16]],
        ) -> pl.Tensor[[L * R, D], pl.BF16]:
            for e in pl.range(L):
                n_rows = pl.cast(pl.read(recv_count, [e, 0]), pl.INDEX)
                for slot in pl.range(n_rows):
                    row = e * R + slot
                    x_bf = pl.load(recv_x_out, [row, 0], [1, D])
                    x_fp = pl.cast(x_bf, target_type=pl.FP32)
                    w_scalar = pl.read(recv_w_out, [e, slot])
                    y_fp = pl.mul(x_fp, w_scalar)
                    y_bf = pl.cast(y_fp, target_type=pl.BF16)
                    pl.store(y_bf, [row, 0], recv_y)
            return recv_y

        # ----------------------------------------------------------------
        # combine — 1:1 of runtime/combine.cpp.
        #
        # Reads recv_idx_out (host-backed [L, R] INT32) and pub_counts
        # (window). Pushes recv_y rows to peer dst's routed_y_buf[r, :] where
        # r = recv_idx_out[e, src_off + row]. combine_done barrier uses
        # per-src signal cells.
        # ----------------------------------------------------------------
        @pl.function(type=pl.FunctionType.InCore)
        def combine_step(
            self,
            recv_y: pl.Tensor[[L * R, D], pl.BF16],
            recv_idx_out: pl.Tensor[[L, R], pl.INT32],
            routed_y_out: pl.Out[pl.Tensor[[T, D], pl.FP32]],
            pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, L], pl.INT32],
            routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
            combine_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[T, D], pl.FP32]:
            # ---------- push: TPUT recv_y rows to peer's routed_y_buf ----
            for dst in pl.range(N_RANKS):
                for e in pl.range(L):
                    n = pl.cast(pl.read(pub_counts, [dst * N_RANKS + my_rank, e]), pl.INDEX)
                    src_off = pl.const(0, pl.INT32)
                    for s in pl.range(N_RANKS):
                        if s < dst:
                            src_off = src_off + pl.read(pub_counts, [s * N_RANKS + my_rank, e])
                    src_off_idx = pl.cast(src_off, pl.INDEX)
                    for row in pl.range(n):
                        idx_lin = e * R + src_off_idx + row
                        r_route = pl.cast(pl.read(recv_idx_out, [e, src_off_idx + row]), pl.INDEX)
                        # HCCL TPUT — load recv_y row + store to peer's routed_y_buf
                        pld.tensor.put(
                            dst=routed_y_buf,
                            peer=dst,
                            src=recv_y,
                            dst_offsets=[r_route, 0],
                            src_offsets=[idx_lin, 0],
                            shape=[1, D],
                        )

            # ---------- combine_done barrier — per-src signal cells ----------
            for peer in pl.range(N_RANKS):
                if peer != my_rank:
                    pld.system.notify(
                        target=combine_done,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(N_RANKS):
                if src != my_rank:
                    pld.system.wait(
                        signal=combine_done,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            # ---------- reduce: routed_y[t] = sum_k cast_fp32(routed_y_buf[t*TOPK+k]) ----
            # FP32 accumulator across TOPK BF16 contributions. Seed `acc` from
            # k=0 then add k=1..TOPK-1 in a loop — works for any TOPK ≥ 1; the
            # SSA pass picks up `acc` as a loop-carried variable.
            for t in pl.range(T):
                y0 = pl.load(routed_y_buf, [t * TOPK, 0], [1, D])
                acc = pl.cast(y0, target_type=pl.FP32)
                for kk in pl.range(TOPK - 1):
                    k = kk + 1
                    y = pl.load(routed_y_buf, [t * TOPK + k, 0], [1, D])
                    y_fp = pl.cast(y, target_type=pl.FP32)
                    acc = pl.add(acc, y_fp)
                pl.store(acc, [t, 0], routed_y_out)
            return routed_y_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(  # noqa: PLR0913
            self,
            indices: pl.Tensor[[T, TOPK], pl.INT32],
            x_norm: pl.Tensor[[T, D], pl.BF16],
            w_padded: pl.Tensor[[N_ROUTES, W_PAD], pl.FP32],
            idx_padded: pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
            recv_x_out: pl.Out[pl.Tensor[[L * R, D], pl.BF16]],
            recv_w_out: pl.Out[pl.Tensor[[L, R], pl.FP32]],
            recv_idx_out: pl.Out[pl.Tensor[[L, R], pl.INT32]],
            recv_count_out: pl.Out[pl.Tensor[[L, 1], pl.INT32]],
            recv_y: pl.Out[pl.Tensor[[L * R, D], pl.BF16]],
            routed_y: pl.Out[pl.Tensor[[T, D], pl.FP32]],
            pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, L], pl.INT32],
            count_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            recv_x: pld.DistributedTensor[[L * R, D], pl.BF16],
            recv_w: pld.DistributedTensor[[L * R, W_PAD], pl.FP32],
            recv_idx: pld.DistributedTensor[[L * R, IDX_PAD], pl.INT32],
            data_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
            combine_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[T, D], pl.FP32]:
            recv_x_out, recv_w_out, recv_idx_out, recv_count_out = self.dispatch_step(
                indices,
                x_norm,
                w_padded,
                idx_padded,
                recv_x_out,
                recv_w_out,
                recv_idx_out,
                recv_count_out,
                pub_counts,
                count_done,
                recv_x,
                recv_w,
                recv_idx,
                data_done,
                my_rank,
            )
            recv_y = self.local_expert_step(recv_x_out, recv_w_out, recv_count_out, recv_y)
            return self.combine_step(
                recv_y,
                recv_idx_out,
                routed_y,
                pub_counts,
                routed_y_buf,
                combine_done,
                my_rank,
            )

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            indices: pl.Tensor[[N_RANKS, T, TOPK], pl.INT32],
            x_norms: pl.Tensor[[N_RANKS, T, D], pl.BF16],
            w_padded: pl.Tensor[[N_RANKS, N_ROUTES, W_PAD], pl.FP32],
            idx_padded: pl.Tensor[[N_RANKS, N_ROUTES, IDX_PAD], pl.INT32],
            recv_x_outs: pl.Out[pl.Tensor[[N_RANKS, L * R, D], pl.BF16]],
            recv_w_outs: pl.Out[pl.Tensor[[N_RANKS, L, R], pl.FP32]],
            recv_idx_outs: pl.Out[pl.Tensor[[N_RANKS, L, R], pl.INT32]],
            recv_count_outs: pl.Out[pl.Tensor[[N_RANKS, L, 1], pl.INT32]],
            recv_ys: pl.Out[pl.Tensor[[N_RANKS, L * R, D], pl.BF16]],
            routed_ys: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.FP32]],
        ) -> pl.Tensor[[N_RANKS, T, D], pl.FP32]:
            # Window allocations — one buffer per cross-rank slot. Barrier
            # signals are sized [N_RANKS, 1] to host per-src cells, matching
            # count_done_sig[N] / data_done_sig[N] / combine_done_sig[N].
            pub_counts_buf = pld.alloc_window_buffer(N_RANKS * N_RANKS * L * 4)  # INT32
            count_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
            recv_x_buf = pld.alloc_window_buffer(L * R * D * 2)  # BF16
            recv_w_buf = pld.alloc_window_buffer(L * R * W_PAD * 4)  # FP32
            recv_idx_buf = pld.alloc_window_buffer(L * R * IDX_PAD * 4)  # INT32
            data_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
            routed_y_buf_buf = pld.alloc_window_buffer(N_ROUTES * D * 2)  # BF16
            combine_done_buf = pld.alloc_window_buffer(N_RANKS * 4)

            for r in pl.range(pld.world_size()):
                pub_counts = pld.window(pub_counts_buf, [N_RANKS * N_RANKS, L], dtype=pl.INT32)
                count_done = pld.window(count_done_buf, [N_RANKS, 1], dtype=pl.INT32)
                recv_x = pld.window(recv_x_buf, [L * R, D], dtype=pl.BF16)
                recv_w = pld.window(recv_w_buf, [L * R, W_PAD], dtype=pl.FP32)
                recv_idx = pld.window(recv_idx_buf, [L * R, IDX_PAD], dtype=pl.INT32)
                data_done = pld.window(data_done_buf, [N_RANKS, 1], dtype=pl.INT32)
                routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
                combine_done = pld.window(combine_done_buf, [N_RANKS, 1], dtype=pl.INT32)
                self.chip_orch(
                    indices[r],
                    x_norms[r],
                    w_padded[r],
                    idx_padded[r],
                    recv_x_outs[r],
                    recv_w_outs[r],
                    recv_idx_outs[r],
                    recv_count_outs[r],
                    recv_ys[r],
                    routed_ys[r],
                    pub_counts,
                    count_done,
                    recv_x,
                    recv_w,
                    recv_idx,
                    data_done,
                    routed_y_buf,
                    combine_done,
                    r,
                    device=r,
                )
            return routed_ys

    return EpDispatchCombine


def _generate_routing_indices(seed: int, n_ranks: int) -> torch.Tensor:
    """Generate ``indices[n_ranks][T, TOPK]`` so no expert exceeds RECV_MAX."""
    e_global = n_ranks * L
    rng = torch.Generator().manual_seed(seed)
    while True:
        indices = torch.zeros(n_ranks, T, TOPK, dtype=torch.int32)
        for r in range(n_ranks):
            for t in range(T):
                perm = torch.randperm(e_global, generator=rng)[:TOPK]
                indices[r, t, :] = perm.to(torch.int32)

        per_expert = torch.zeros(n_ranks, L, dtype=torch.int32)
        for r in range(n_ranks):
            for t in range(T):
                for k in range(TOPK):
                    eid = int(indices[r, t, k].item())
                    dst = eid // L
                    loc_e = eid % L
                    per_expert[dst, loc_e] += 1
        if int(per_expert.max().item()) <= R:
            return indices
        seed += 1
        rng.manual_seed(seed)


def _pack_weights_padded(weights: torch.Tensor) -> torch.Tensor:
    n_ranks = weights.shape[0]
    out = torch.zeros((n_ranks, N_ROUTES, W_PAD), dtype=torch.float32)
    for r in range(n_ranks):
        for t in range(T):
            for k in range(TOPK):
                r_route = t * TOPK + k
                out[r, r_route, 0] = weights[r, t, k]
    return out


def _pack_idx_padded(n_ranks: int) -> torch.Tensor:
    out = torch.zeros((n_ranks, N_ROUTES, IDX_PAD), dtype=torch.int32)
    for t in range(T):
        for k in range(TOPK):
            r_route = t * TOPK + k
            out[:, r_route, 0] = r_route
    return out


def _compute_golden_recv(
    x_norms: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay dispatch protocol on host → per-rank dispatch goldens.

    Mirrors runtime ``compute_golden`` in main.py:
      expected_recv_x[r]   BF16  [L, R, D]
      expected_recv_w[r]   FP32  [L, R]
      expected_recv_idx[r] INT32 [L, R]   (r_route = t * TOPK + k)
      expected_count[r]    INT32 [L]
    """
    n_ranks = x_norms.shape[0]
    expected_recv_x = torch.zeros(n_ranks, L, R, D, dtype=torch.bfloat16)
    expected_recv_w = torch.zeros(n_ranks, L, R, dtype=torch.float32)
    expected_recv_idx = torch.zeros(n_ranks, L, R, dtype=torch.int32)
    expected_count = torch.zeros(n_ranks, L, dtype=torch.int32)

    send_counts = torch.zeros(n_ranks, n_ranks, L, dtype=torch.int32)
    for src in range(n_ranks):
        for t in range(T):
            for k in range(TOPK):
                eid = int(indices[src, t, k].item())
                dst = eid // L
                loc_e = eid % L
                send_counts[src, dst, loc_e] += 1

    for dst in range(n_ranks):
        slot_offset = torch.zeros(n_ranks, L, dtype=torch.int32)
        running = torch.zeros(L, dtype=torch.int32)
        for src in range(n_ranks):
            slot_offset[src] = running.clone()
            running = running + send_counts[src, dst]

        for src in range(n_ranks):
            cursor = torch.zeros(L, dtype=torch.int32)
            for t in range(T):
                for k in range(TOPK):
                    eid = int(indices[src, t, k].item())
                    if eid // L != dst:
                        continue
                    loc_e = eid % L
                    slot = int(slot_offset[src, loc_e].item() + cursor[loc_e].item())
                    cursor[loc_e] += 1
                    expected_recv_x[dst, loc_e, slot, :] = x_norms[src, t, :]
                    expected_recv_w[dst, loc_e, slot] = weights[src, t, k]
                    expected_recv_idx[dst, loc_e, slot] = t * TOPK + k

        for e in range(L):
            expected_count[dst, e] = int(running[e].item())

    return expected_recv_x, expected_recv_w, expected_recv_idx, expected_count


def _compute_golden_routed(x_norms: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Per-rank ``routed_y`` golden — only depends on r's own inputs because
    the dispatch+combine protocol is end-to-end shape-preserving for routed_y
    (each (t, k) on rank r round-trips back to the original (t, k) slot).
    """
    n_ranks = x_norms.shape[0]
    expected = torch.zeros((n_ranks, T, D), dtype=torch.float32)
    for r in range(n_ranks):
        for t in range(T):
            for k in range(TOPK):
                weighted = float(weights[r, t, k].item()) * x_norms[r, t, :].to(torch.float32)
                expected[r, t, :] += weighted.to(torch.bfloat16).to(torch.float32)
    return expected


class TestL3EpDispatchCombine:
    """L3 distributed runtime: N-rank EP dispatch + local_expert + combine."""

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_ep_dispatch_combine(self, test_config, device_ids, n_ranks):
        if len(device_ids) < n_ranks:
            pytest.skip(f"ep_dispatch_combine needs {n_ranks} devices, got {device_ids}")

        program = _build_ep_dispatch_combine_program(n_ranks)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        x_norms = torch.tensor(
            [[[r * 100 + t * 10 + d for d in range(D)] for t in range(T)] for r in range(n_ranks)],
            dtype=torch.bfloat16,
        )
        weights = torch.tensor(
            [
                [[(r + 1) * 0.01 + t * 0.1 + k * 0.001 for k in range(TOPK)] for t in range(T)]
                for r in range(n_ranks)
            ],
            dtype=torch.float32,
        )
        indices = _generate_routing_indices(seed=20260510, n_ranks=n_ranks)
        weights_padded = _pack_weights_padded(weights)
        idx_padded = _pack_idx_padded(n_ranks)

        recv_x_outs = torch.zeros((n_ranks, L * R, D), dtype=torch.bfloat16)
        recv_w_outs = torch.zeros((n_ranks, L, R), dtype=torch.float32)
        recv_idx_outs = torch.zeros((n_ranks, L, R), dtype=torch.int32)
        recv_count_outs = torch.zeros((n_ranks, L, 1), dtype=torch.int32)
        recv_ys = torch.zeros((n_ranks, L * R, D), dtype=torch.bfloat16)
        routed_ys = torch.zeros((n_ranks, T, D), dtype=torch.float32)

        compiled(
            indices,
            x_norms,
            weights_padded,
            idx_padded,
            recv_x_outs,
            recv_w_outs,
            recv_idx_outs,
            recv_count_outs,
            recv_ys,
            routed_ys,
        )

        # ---------- dispatch-stage goldens ----------
        expected_recv_x, expected_recv_w, expected_recv_idx, expected_count = _compute_golden_recv(
            x_norms, indices, weights
        )
        recv_count_outs_2d = recv_count_outs.squeeze(-1)
        assert torch.equal(recv_count_outs_2d, expected_count), (
            f"recv_count mismatch: got={recv_count_outs_2d.tolist()} expected={expected_count.tolist()}"
        )
        recv_x_outs_4d = recv_x_outs.reshape(n_ranks, L, R, D)
        for r in range(n_ranks):
            for e in range(L):
                n = int(expected_count[r, e].item())
                if n == 0:
                    continue
                got_x = recv_x_outs_4d[r, e, :n, :].to(torch.float32)
                exp_x = expected_recv_x[r, e, :n, :].to(torch.float32)
                assert torch.equal(got_x, exp_x), (
                    f"recv_x mismatch at rank {r} expert {e}: max diff = {(got_x - exp_x).abs().max().item()}"
                )
                got_w = recv_w_outs[r, e, :n]
                exp_w = expected_recv_w[r, e, :n]
                assert torch.allclose(got_w, exp_w, atol=1e-6), (
                    f"recv_w mismatch at rank {r} expert {e}: max diff = {(got_w - exp_w).abs().max().item()}"
                )
                got_idx = recv_idx_outs[r, e, :n]
                exp_idx = expected_recv_idx[r, e, :n]
                assert torch.equal(got_idx, exp_idx), (
                    f"recv_idx mismatch at rank {r} expert {e}: "
                    f"got={got_idx.tolist()} expected={exp_idx.tolist()}"
                )

        # ---------- combine-stage golden ----------
        # routed_y[t, :] = sum_k bf16(weights[t,k] * x_norms[t]) accumulated in
        # FP32 — each BF16 cast can differ from torch's round-to-nearest-even
        # by 1 ULP on an exact tie, and summing TOPK terms keeps the *relative*
        # error within ~2 BF16 ULPs (FP32 accumulation is exact). 2**-7 is one
        # BF16 ULP relative; rtol = 2**-6 admits 2-ULP slack. Mirrors the
        # runtime example's `_verify_routed_y` tolerance.
        expected_routed = _compute_golden_routed(x_norms, weights)
        max_diff = (routed_ys - expected_routed).abs().max().item()
        ulp_2 = 2.0**-6
        assert torch.allclose(routed_ys, expected_routed, atol=ulp_2, rtol=ulp_2), (
            f"ep_dispatch_combine routed_y mismatch: max diff = {max_diff}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
