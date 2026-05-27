# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: 2-rank EP dispatch + local_expert + combine — 1:1 PyPTO
port of ``runtime/examples/workers/l3/ep_dispatch_combine``.

This is a structural port of the C++ runtime example, not a simplified analog.
The three AIV kernels (``dispatch`` / ``local_expert`` / ``combine``) become
three :class:`pl.FunctionType.InCore` kernels chained from ``chip_orch``, with
the same window layout, the same routing protocol, and the same data
direction at every cross-rank op:

* **dispatch**: histogram → publish ``send_counts`` via TNOTIFY(AtomicAdd) +
  count_done barrier → prefix_sum → ``payload_push`` 3-channel push (x BF16 /
  w FP32 / idx INT32) into peer's ``recv_x``/``recv_w``/``recv_idx`` keyed by
  ``(local_expert, slot)`` → data_done barrier.
* **local_expert**: ``recv_y[e, s, :] = cast_bf16(cast_fp32(recv_x) *
  recv_w[..., 0])`` with the BF16 round-trip preserved.
* **combine**: TPUT-style push of ``recv_y[idx_lin, :]`` to peer's
  ``routed_y_buf[r, :]`` where ``r = t * TOPK + k`` from ``recv_idx``, then
  combine_done barrier, then FP32 reduce_sum along TOPK into ``routed_y``.

The cross-rank push points (``payload_push`` in dispatch, the recv_y push in
combine) use the new :func:`pld.tile.remote_store` op. The TPUT semantics
decompose as ``local_load + remote_subview_store``: each push site is one
``pl.load`` of a local tile followed by one ``pld.tile.remote_store`` to a
subview of a peer's window-bound tensor. No data-direction inversion.

The histogram phase reads ``indices`` via ``pl.read`` scalar GM accesses
(matching the runtime kernel's ``int eid = indices[r];`` pattern) so the
``[T, TOPK]`` INT32 tensor doesn't need to be padded to a 32-byte-aligned
vector tile width; ``pl.array.create`` carries ``send_counts`` / ``cursor``
register-local arrays so the scalar control flow translates directly.

Sort vs. natural order — the runtime kernel insertion-sorts routes by
``(dst, loc_e)`` so the payload_push cursor advances within each bucket
contiguously. Slot determinism only requires that ``cursor[dst][loc_e]`` be
incremented exactly once per route hitting that bucket, so any traversal
order works; we iterate ``(t, k)`` in natural lexicographic order, which is
simpler to express in pypto and produces the same delivered routes (just at
different slot offsets within each src's slab).

The 2-rank constraint matches the runtime example exactly; for ``N>2`` the
publish/barrier loops would generalize to per-src signal cells, mirroring
``count_done_sig[N]`` in the C++ kernel.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

# Demo dimensions — must mirror the runtime example's constants.
N_RANKS = 2
T = 8
TOPK = 2
D = 64
L = 4  # N_LOCAL_EXPERTS per rank
R = 32  # RECV_MAX (single-expert receive upper bound)
W_PAD = 8  # weight tile width — minimum vector tile (1x8 FP32 = 32 B)
IDX_PAD = 8  # idx tile width   — minimum vector tile (1x8 INT32 = 32 B)
E_GLOBAL = N_RANKS * L
N_ROUTES = T * TOPK  # 16


def _build_ep_dispatch_combine_program():
    """Build the 2-rank ep_dispatch_combine program at call time.

    Deferred construction matches other L3 tests — keeps the module importable
    even if the embedded body trips the parser at collection time.
    """

    @pl.program
    class EpDispatchCombine:
        # ----------------------------------------------------------------
        # dispatch — 1:1 of runtime/dispatch.cpp.
        #
        # Reads:   indices, x_norm, w_padded, idx_padded (host-backed inputs)
        # Writes:  recv_count_out (host-backed [L, 1] INT32 output)
        #          pub_counts, recv_x, recv_w, recv_idx (window slots)
        # Barriers: count_done (publish→prefix_sum), data_done (push→exit)
        # ----------------------------------------------------------------
        @pl.function(type=pl.FunctionType.InCore)
        def dispatch_step(  # noqa: PLR0913, PLR0912
            self,
            indices: pl.Tensor[[T, TOPK], pl.INT32],
            x_norm: pl.Tensor[[T, D], pl.BF16],
            w_padded: pl.Tensor[[N_ROUTES, W_PAD], pl.FP32],
            idx_padded: pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
            recv_count_out: pl.Out[pl.Tensor[[L, 1], pl.INT32]],
            pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, L], pl.INT32],
            count_done: pld.DistributedTensor[[1, 1], pl.INT32],
            recv_x: pld.DistributedTensor[[L * R, D], pl.BF16],
            recv_w: pld.DistributedTensor[[L * R, W_PAD], pl.FP32],
            recv_idx: pld.DistributedTensor[[L * R, IDX_PAD], pl.INT32],
            data_done: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[L, 1], pl.INT32]:
            # ---------- histogram: scalar histogram on indices ----------
            # Read each (t, k) route via scalar GM access, increment
            # send_counts[d][e]. send_counts is a register-local INT32 array
            # of length N_RANKS*L — same shape as the runtime kernel's
            # ``int send_counts[N][L]`` C-stack array.
            send_counts = pl.array.create(N_RANKS * L, pl.INT32)
            for d in pl.unroll(N_RANKS):
                for e in pl.unroll(L):
                    send_counts[d * L + e] = 0

            for t in pl.unroll(T):
                for k in pl.unroll(TOPK):
                    eid = pl.read(indices, [t, k])
                    d = eid // L
                    e = eid - d * L
                    cur = send_counts[d * L + e]
                    send_counts[d * L + e] = cur + 1

            # ---------- publish: TNOTIFY(AtomicAdd) send_counts to peers ----
            # Each rank publishes its full [N_RANKS, L] send_counts row to
            # every peer's pub_counts[my_rank][:][:] slice. AtomicAdd from
            # zero is equivalent to a store (HCCL window zero-init), and
            # self-rank is included so pub_counts[my_rank][:][:] gets
            # populated locally too.
            for peer_const in pl.unroll(N_RANKS):
                for d in pl.unroll(N_RANKS):
                    for e in pl.unroll(L):
                        v = send_counts[d * L + e]
                        # Skip v == 0 cells — matches runtime/dispatch.cpp's
                        # `if (v == 0) continue;` (AtomicAdd 0 is a no-op but
                        # still issues a cross-rank op).
                        if v != 0:
                            # Flatten (my_rank, d, e) into the 2D pub_counts
                            # layout: row = my_rank * N_RANKS + d, col = e.
                            pld.system.notify(
                                target=pub_counts,
                                peer=peer_const,
                                offsets=[my_rank * N_RANKS + d, e],
                                value=v,
                                op=pld.NotifyOp.AtomicAdd,
                            )

            # ---------- count_done barrier ----------
            pld.system.notify(
                target=count_done,
                peer=peer,
                offsets=[0, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
            pld.system.wait(
                signal=count_done,
                offsets=[0, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )

            # ---------- prefix_sum: my_slot_at_dst + recv_count ----------
            # my_slot_at_dst[dst][e] = sum_{s<my_rank} pub_counts[s][dst][e]
            #   — sender's slot offset on each peer's recv area.
            # recv_count[e]        = sum_{s<N_RANKS} pub_counts[s][my_rank][e]
            #   — total rows arriving at THIS rank's local expert e.
            my_slot_at_dst = pl.array.create(N_RANKS * L, pl.INT32)
            for d in pl.unroll(N_RANKS):
                for e in pl.unroll(L):
                    acc = pl.array.create(1, pl.INT32)
                    acc[0] = 0
                    for s in pl.unroll(N_RANKS):
                        if s < my_rank:
                            acc[0] = acc[0] + pl.read(pub_counts, [s * N_RANKS + d, e])
                    my_slot_at_dst[d * L + e] = acc[0]

            for e in pl.unroll(L):
                acc = pl.array.create(1, pl.INT32)
                acc[0] = 0
                for s in pl.unroll(N_RANKS):
                    acc[0] = acc[0] + pl.read(pub_counts, [s * N_RANKS + my_rank, e])
                pl.write(recv_count_out, [e, 0], acc[0])

            # ---------- payload_push: 3-channel push via remote_store ------
            # For each route (t, k), look up (dst, loc_e) from indices,
            # compute slot = my_slot_at_dst[dst][loc_e] + cursor[dst][loc_e],
            # and push x / w / idx as 1×C tiles to peer's recv_x/w/idx at
            # row = loc_e * R + slot.
            #
            # Self-rank is NOT skipped: dst can equal my_rank for tokens
            # that route to a local expert. CommRemotePtr returns the local
            # address for peer==my_rank, so remote_store falls back to a
            # local subview store automatically.
            cursor = pl.array.create(N_RANKS * L, pl.INT32)
            for d in pl.unroll(N_RANKS):
                for e in pl.unroll(L):
                    cursor[d * L + e] = 0

            for t in pl.unroll(T):
                # x_norm[t, :] is reused across both k iterations — load once
                # per t and reuse for every (t, k) push.
                x_tile = pl.load(x_norm, [t, 0], [1, D])
                for k in pl.unroll(TOPK):
                    eid = pl.read(indices, [t, k])
                    dst = eid // L
                    loc_e = eid - dst * L
                    bucket = dst * L + loc_e
                    cur_val = cursor[bucket]
                    cursor[bucket] = cur_val + 1
                    slot_off = my_slot_at_dst[bucket]
                    slot = slot_off + cur_val
                    row = loc_e * R + slot

                    # Channel 1: x BF16 [1, D] — x_norm[t, :] → peer.recv_x[row, :]
                    pld.tile.remote_store(x_tile, target=recv_x, peer=dst, offsets=[row, 0])

                    # Channel 2: w FP32 [1, W_PAD] — host pre-packed [w, 0, ..., 0]
                    r_route = t * TOPK + k
                    w_tile = pl.load(w_padded, [r_route, 0], [1, W_PAD])
                    pld.tile.remote_store(w_tile, target=recv_w, peer=dst, offsets=[row, 0])

                    # Channel 3: idx INT32 [1, IDX_PAD] — host pre-packed [r, 0, ..., 0]
                    idx_tile = pl.load(idx_padded, [r_route, 0], [1, IDX_PAD])
                    pld.tile.remote_store(idx_tile, target=recv_idx, peer=dst, offsets=[row, 0])

            # ---------- data_done barrier ----------
            pld.system.notify(
                target=data_done,
                peer=peer,
                offsets=[0, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
            pld.system.wait(
                signal=data_done,
                offsets=[0, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )
            return recv_count_out

        # ----------------------------------------------------------------
        # local_expert — 1:1 of runtime/local_expert.cpp.
        #
        # recv_y[e, slot, :] = cast_bf16(cast_fp32(recv_x[e, slot, :]) *
        #                                recv_w[e, slot, 0])    for slot < recv_count[e]
        #
        # Pure local — no cross-rank ops.
        # ----------------------------------------------------------------
        @pl.function(type=pl.FunctionType.InCore)
        def local_expert_step(
            self,
            recv_count: pl.Tensor[[L, 1], pl.INT32],
            recv_y: pl.Out[pl.Tensor[[L * R, D], pl.BF16]],
            recv_x: pld.DistributedTensor[[L * R, D], pl.BF16],
            recv_w: pld.DistributedTensor[[L * R, W_PAD], pl.FP32],
        ) -> pl.Tensor[[L * R, D], pl.BF16]:
            for e in pl.unroll(L):
                # pl.read returns the tensor's INT32 dtype; pl.range needs an
                # INDEX bound, so cast explicitly (matches the row_idx cast
                # idiom in tests/st/runtime/.../test_incore_array.py).
                n_rows = pl.cast(pl.read(recv_count, [e, 0]), pl.INDEX)
                for slot in pl.range(n_rows):
                    row = e * R + slot
                    x_bf = pl.load(recv_x, [row, 0], [1, D])
                    x_fp = pl.cast(x_bf, target_type=pl.FP32)
                    w_scalar = pl.read(recv_w, [row, 0])
                    y_fp = pl.mul(x_fp, w_scalar)
                    y_bf = pl.cast(y_fp, target_type=pl.BF16)
                    pl.store(y_bf, [row, 0], recv_y)
            return recv_y

        # ----------------------------------------------------------------
        # combine — 1:1 of runtime/combine.cpp.
        #
        # Phase push: for each (dst, e), push recv_y rows to peer dst's
        #             routed_y_buf[r, :] (r = t*TOPK+k from recv_idx).
        # combine_done barrier.
        # Phase reduce: for each token t, reduce_sum cast_fp32(
        #               routed_y_buf[t*TOPK+k, :]) over k into routed_y[t, :] FP32.
        #
        # The push direction is the same as runtime — recv_y on rank A
        # carries OUR tokens' expert outputs (A ran the expert step on data
        # we pushed via dispatch), so we push them BACK to our originating
        # ranks' routed_y_buf. With remote_store, this is one pl.load of a
        # 1×D BF16 tile + one pld.tile.remote_store per row.
        # ----------------------------------------------------------------
        @pl.function(type=pl.FunctionType.InCore)
        def combine_step(
            self,
            recv_y: pl.Tensor[[L * R, D], pl.BF16],
            routed_y_out: pl.Out[pl.Tensor[[T, D], pl.FP32]],
            pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, L], pl.INT32],
            recv_idx: pld.DistributedTensor[[L * R, IDX_PAD], pl.INT32],
            routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
            combine_done: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[T, D], pl.FP32]:
            # ---------- push: TPUT recv_y rows to peer's routed_y_buf ----
            for dst in pl.unroll(N_RANKS):
                for e in pl.unroll(L):
                    # n / src_off feed pl.range and offset arithmetic — both
                    # want INDEX. Read as the tensor's INT32 dtype and cast.
                    n = pl.cast(pl.read(pub_counts, [dst * N_RANKS + my_rank, e]), pl.INDEX)
                    # src_off = sum_{s<dst} pub_counts[s][my_rank][e]
                    src_off = pl.array.create(1, pl.INT32)
                    src_off[0] = 0
                    for s in pl.unroll(N_RANKS):
                        if s < dst:
                            src_off[0] = src_off[0] + pl.read(pub_counts, [s * N_RANKS + my_rank, e])
                    src_off_idx = pl.cast(src_off[0], pl.INDEX)
                    for row in pl.range(n):
                        idx_lin = e * R + src_off_idx + row
                        r_route = pl.read(recv_idx, [idx_lin, 0])
                        y_tile = pl.load(recv_y, [idx_lin, 0], [1, D])
                        pld.tile.remote_store(y_tile, target=routed_y_buf, peer=dst, offsets=[r_route, 0])

            # ---------- combine_done barrier ----------
            pld.system.notify(
                target=combine_done,
                peer=peer,
                offsets=[0, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
            pld.system.wait(
                signal=combine_done,
                offsets=[0, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )

            # ---------- reduce: routed_y[t] = sum_k cast_fp32(routed_y_buf[t*TOPK+k]) ----
            # TOPK=2 statically — straight unrolled add, matching the
            # existing combine_step structure (avoids nesting init_values
            # inside the t-loop).
            for t in pl.unroll(T):
                y0 = pl.load(routed_y_buf, [t * TOPK, 0], [1, D])
                y1 = pl.load(routed_y_buf, [t * TOPK + 1, 0], [1, D])
                y0_fp = pl.cast(y0, target_type=pl.FP32)
                y1_fp = pl.cast(y1, target_type=pl.FP32)
                acc = pl.add(y0_fp, y1_fp)
                pl.store(acc, [t, 0], routed_y_out)
            return routed_y_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(  # noqa: PLR0913
            self,
            indices: pl.Tensor[[T, TOPK], pl.INT32],
            x_norm: pl.Tensor[[T, D], pl.BF16],
            w_padded: pl.Tensor[[N_ROUTES, W_PAD], pl.FP32],
            idx_padded: pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
            recv_count_out: pl.Out[pl.Tensor[[L, 1], pl.INT32]],
            recv_y: pl.Out[pl.Tensor[[L * R, D], pl.BF16]],
            routed_y: pl.Out[pl.Tensor[[T, D], pl.FP32]],
            pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, L], pl.INT32],
            count_done: pld.DistributedTensor[[1, 1], pl.INT32],
            recv_x: pld.DistributedTensor[[L * R, D], pl.BF16],
            recv_w: pld.DistributedTensor[[L * R, W_PAD], pl.FP32],
            recv_idx: pld.DistributedTensor[[L * R, IDX_PAD], pl.INT32],
            data_done: pld.DistributedTensor[[1, 1], pl.INT32],
            routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
            combine_done: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[T, D], pl.FP32]:
            # Sequential chaining: dispatch → local_expert → combine.
            # Cross-kernel dependencies flow via host-backed pl.Out outputs
            # (recv_count_out / recv_y / routed_y) and shared window slots.
            # InOut/Out parameters use SSA-functional discipline — rebind via
            # the call's return value before passing to the next kernel.
            recv_count_out = self.dispatch_step(
                indices,
                x_norm,
                w_padded,
                idx_padded,
                recv_count_out,
                pub_counts,
                count_done,
                recv_x,
                recv_w,
                recv_idx,
                data_done,
                peer,
                my_rank,
            )
            recv_y = self.local_expert_step(recv_count_out, recv_y, recv_x, recv_w)
            return self.combine_step(
                recv_y,
                routed_y,
                pub_counts,
                recv_idx,
                routed_y_buf,
                combine_done,
                peer,
                my_rank,
            )

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            indices: pl.Tensor[[N_RANKS, T, TOPK], pl.INT32],
            x_norms: pl.Tensor[[N_RANKS, T, D], pl.BF16],
            w_padded: pl.Tensor[[N_RANKS, N_ROUTES, W_PAD], pl.FP32],
            idx_padded: pl.Tensor[[N_RANKS, N_ROUTES, IDX_PAD], pl.INT32],
            recv_count_outs: pl.Out[pl.Tensor[[N_RANKS, L, 1], pl.INT32]],
            recv_ys: pl.Out[pl.Tensor[[N_RANKS, L * R, D], pl.BF16]],
            routed_ys: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.FP32]],
        ) -> pl.Tensor[[N_RANKS, T, D], pl.FP32]:
            # Window allocations — one buffer per cross-rank slot. Bytes
            # mirror the runtime example's k*Bytes constants.
            pub_counts_buf = pld.alloc_window_buffer(N_RANKS * N_RANKS * L * 4)  # INT32
            count_done_buf = pld.alloc_window_buffer(4)
            recv_x_buf = pld.alloc_window_buffer(L * R * D * 2)  # BF16
            recv_w_buf = pld.alloc_window_buffer(L * R * W_PAD * 4)  # FP32
            recv_idx_buf = pld.alloc_window_buffer(L * R * IDX_PAD * 4)  # INT32
            data_done_buf = pld.alloc_window_buffer(4)
            routed_y_buf_buf = pld.alloc_window_buffer(N_ROUTES * D * 2)  # BF16
            combine_done_buf = pld.alloc_window_buffer(4)

            for r in pl.range(pld.world_size()):
                pub_counts = pld.window(pub_counts_buf, [N_RANKS * N_RANKS, L], dtype=pl.INT32)
                count_done = pld.window(count_done_buf, [1, 1], dtype=pl.INT32)
                recv_x = pld.window(recv_x_buf, [L * R, D], dtype=pl.BF16)
                recv_w = pld.window(recv_w_buf, [L * R, W_PAD], dtype=pl.FP32)
                recv_idx = pld.window(recv_idx_buf, [L * R, IDX_PAD], dtype=pl.INT32)
                data_done = pld.window(data_done_buf, [1, 1], dtype=pl.INT32)
                routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
                combine_done = pld.window(combine_done_buf, [1, 1], dtype=pl.INT32)
                # Ring partner: peer = (r + 1) % nranks; for N_RANKS=2 this
                # is the other rank.
                self.chip_orch(
                    indices[r],
                    x_norms[r],
                    w_padded[r],
                    idx_padded[r],
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
                    (r + 1) % pld.world_size(),
                    r,
                    device=r,
                )
            return routed_ys

    return EpDispatchCombine


def _generate_routing_indices(seed: int) -> torch.Tensor:
    """Generate ``indices[N_RANKS][T, TOPK]`` so no expert exceeds RECV_MAX.

    Top-k entries within a single token are forced unique. Reseed if any
    per-expert receive count would overflow R. Mirrors the runtime example's
    ``generate_routing_indices``.
    """
    rng = torch.Generator().manual_seed(seed)
    while True:
        indices = torch.zeros(N_RANKS, T, TOPK, dtype=torch.int32)
        for r in range(N_RANKS):
            for t in range(T):
                perm = torch.randperm(E_GLOBAL, generator=rng)[:TOPK]
                indices[r, t, :] = perm.to(torch.int32)

        per_expert = torch.zeros(N_RANKS, L, dtype=torch.int32)
        for r in range(N_RANKS):
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
    """Pack ``[N_RANKS, T, TOPK]`` weights into ``[N_RANKS, N_ROUTES, W_PAD]`` FP32.

    Mirrors the runtime example's ``pack_weights_padded``: row r=t*TOPK+k is
    ``[weight_value, 0, ..., 0]`` — actual weight at column 0, zeros at
    [1, W_PAD). The receiver writes the full 1×W_PAD tile and reads column 0.
    """
    out = torch.zeros((N_RANKS, N_ROUTES, W_PAD), dtype=torch.float32)
    for r in range(N_RANKS):
        for t in range(T):
            for k in range(TOPK):
                r_route = t * TOPK + k
                out[r, r_route, 0] = weights[r, t, k]
    return out


def _pack_idx_padded() -> torch.Tensor:
    """Pack ``[N_RANKS, N_ROUTES, IDX_PAD]`` idx tiles where row r = (r, 0, ..., 0).

    Identical layout for every rank — r = t*TOPK + k is intrinsic, not
    rank-specific. Mirrors the runtime example's ``pack_idx_padded``.
    """
    out = torch.zeros((N_RANKS, N_ROUTES, IDX_PAD), dtype=torch.int32)
    for t in range(T):
        for k in range(TOPK):
            r_route = t * TOPK + k
            out[:, r_route, 0] = r_route
    return out


def _compute_golden(x_norms: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Host reference for ``routed_y`` — mirrors the runtime ``_verify_routed_y``.

    For each rank r: ``routed_y[r][t, :] = sum_k cast_fp32(cast_bf16(
    x_norms[r][t, :].fp32 * weights[r, t, k]))``.

    The dispatch protocol is end-to-end shape-preserving for routed_y: each
    (t, k) on rank r dispatches to some (dst, loc_e), gets multiplied by the
    rank's weight, cast through BF16, and pushed back by combine to the
    original (t, k) slot on rank r. So the formula depends only on r's own
    inputs — routing details cancel out. Takes the *unpadded* ``[N_RANKS,
    T, TOPK]`` weights — the W_PAD layout only affects on-device tile width
    and never reaches the reduce.
    """
    expected = torch.zeros((N_RANKS, T, D), dtype=torch.float32)
    for r in range(N_RANKS):
        for t in range(T):
            for k in range(TOPK):
                weighted = float(weights[r, t, k].item()) * x_norms[r, t, :].to(torch.float32)
                expected[r, t, :] += weighted.to(torch.bfloat16).to(torch.float32)
    return expected


class TestL3EpDispatchCombine:
    """L3 distributed runtime: 2-rank EP dispatch + local_expert + combine."""

    def test_ep_dispatch_combine(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"ep_dispatch_combine needs 2 devices, got {device_ids}")
        if test_config.platform.endswith("sim"):
            pytest.skip(
                "PTOAS auto-emits a tail pipe_barrier+dcci+dsb block outside the "
                "__DAV_VEC__ guard for kernels of this size/shape; sim builds "
                "leave dcci/dsb/ENTIRE_DATA_CACHE undefined at file scope (they "
                "live in inner_kernel.h, which isn't transitively included from "
                "pto-inst.hpp). Re-enable once either PTOAS guards both dcci "
                "blocks or inner_kernel.h is included unconditionally."
            )

        program = _build_ep_dispatch_combine_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

        # x_norm[r, t, d] = r*100 + t*10 + d  → max = 1*100 + 7*10 + 63 = 233.
        # All values are integers ≤ 256 so they fit BF16 exactly (8-bit
        # mantissa + hidden bit gives exact integers up to 2^8). The host
        # golden therefore lines up bit-for-bit on the BF16 round-trip.
        x_norms = torch.tensor(
            [[[r * 100 + t * 10 + d for d in range(D)] for t in range(T)] for r in range(N_RANKS)],
            dtype=torch.bfloat16,
        )
        weights = torch.tensor(
            [
                [[(r + 1) * 0.01 + t * 0.1 + k * 0.001 for k in range(TOPK)] for t in range(T)]
                for r in range(N_RANKS)
            ],
            dtype=torch.float32,
        )
        indices = _generate_routing_indices(seed=20260510)
        weights_padded = _pack_weights_padded(weights)
        idx_padded = _pack_idx_padded()

        # Host-backed intermediates that need pre-allocation. recv_count_out
        # and recv_ys are kernel outputs that combine reads; they need to be
        # passed to the orch as OUTPUT_EXISTING tensors.
        recv_count_outs = torch.zeros((N_RANKS, L, 1), dtype=torch.int32)
        recv_ys = torch.zeros((N_RANKS, L * R, D), dtype=torch.bfloat16)
        routed_ys = torch.zeros((N_RANKS, T, D), dtype=torch.float32)

        compiled(
            indices,
            x_norms,
            weights_padded,
            idx_padded,
            recv_count_outs,
            recv_ys,
            routed_ys,
        )

        expected = _compute_golden(x_norms, indices, weights)
        max_diff = (routed_ys - expected).abs().max().item()
        # 1e-3 mirrors the runtime example's tolerance — the only error
        # source is the per-(t, k) BF16 cast that both sides perform
        # identically.
        assert torch.allclose(routed_ys, expected, atol=1e-3), (
            f"ep_dispatch_combine mismatch: max diff = {max_diff}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
