# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST: variable-size all-to-all via ``pld.tensor.all_to_all_v`` intrinsic.

Validates the variable-size composite all-to-all intrinsic (MPI_Alltoallv pattern)
produces the correct rank-ordered personalized exchange, and that ``recv_counts``
publishes the receive-side counts (no hardcoded ``n_rows = nr - rank`` on device).

The intrinsic takes five arguments with flat 2D layouts for ptoas compatibility:
  - ``input`` (Tensor [NR*MAX_RECV, SIZE]) — per-destination chunks
  - ``target`` (DistributedTensor [NR*MAX_RECV, SIZE]) — flat 2D staging window
  - ``signal`` (DistributedTensor INT32 [NR, 1]) — barrier
  - ``send_counts`` (Tensor INT32 [NR, 1]) — rows to send to each destination,
    read at runtime and clamped to MAX_RECV
  - ``recv_counts`` (DistributedTensor INT32 [NR, 1]) — after the barrier,
    ``recv_counts[src, 0]`` holds how many rows ``src`` sent here (MPI_Alltoallv
    recvcounts, published via ``pld.system.notify``), so the receiver can skip
    unwritten holes without hardcoding

Window-as-result pattern: the intrinsic returns the target window, and the caller
reads back with ``pl.load`` — exactly the same pattern as the symmetric
``pld.tensor.all_to_all``.

The TPUT engine transfers exactly the rows being sent — the transfer shape is
[rows, SIZE], where ``rows = clamp(send_counts[dest], 0, MAX_RECV)`` is read at
runtime — using a compact [1, SIZE] staging tile that it auto-chunks through.
PTOAS accepts dynamic partition-view dims on ``pto.comm.tput``, so the padding
up to MAX_RECV never crosses the interconnect. The receiver uses the published
recv_counts to identify valid rows and skip unwritten window holes.

ST coverage: **P=2** (default CI / 2-device hosts) and **P=4** (any four devices).
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir import DistributedConfig

SIZE = 64
MAX_RECV = 4


def _build_all_to_all_v_program(n_ranks: int, max_recv: int):
    """Build an N-rank variable-size all-to-all program."""
    nr = n_ranks
    mr = max_recv
    total = nr * mr

    @pl.program
    class AllToAllVIntrinsicNRank:
        """N-rank variable-size all-to-all program with window-as-result pattern."""

        @pl.function(type=pl.FunctionType.InCore)
        def exchange_step(
            self,
            inp: pl.Tensor[[total, SIZE], pl.FP32],
            counts: pl.Tensor[[nr, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[total, SIZE], pl.FP32]],
            recv_out: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
            data: pl.InOut[pld.DistributedTensor[[total, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            recv_counts: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[total, SIZE], pl.FP32], pl.Tensor[[nr, 1], pl.INT32]]:
            """InCore kernel: push variable rows per peer, barrier, read back via recv_counts."""
            # Push-based all_to_all_v — intrinsic pushes counts[dest] rows to
            # each peer, publishes counts into recv_counts, and returns data
            # in-place (window-as-result).
            result = pld.tensor.all_to_all_v(inp, data, signal, counts, recv_counts)
            # Read back valid rows from the window for host-side verification.
            # Loop bounded by published recv_counts[src] (not a hardcoded formula
            # / MAX_RECV). The TPUT transfer shape is the runtime [rows, SIZE]
            # (PTOAS accepts dynamic partition-view dims on pto.comm.tput); a
            # [1, SIZE] staging tile feeds the engine. The receiver uses
            # recv_counts to skip unwritten window holes.
            # Both loops target `out`, and their row ranges partition each
            # sender's slot, so every row of the window is copied exactly once
            # and `out` ends up FULLY written. That last part matters: a
            # `pl.Out` tensor is write-only on the device (the host buffer is
            # never uploaded), so any row the kernel skipped would come back as
            # undefined memory rather than as window content.
            #   [base, base + recv_counts[src])       valid -- checked vs golden
            #   [base + recv_counts[src], base + mr)  tail  -- bounded-transfer check
            # Keeping the valid-row loop bounded by the published recv_counts
            # (not a hardcoded formula / MAX_RECV) exercises the intended
            # device-side consumer pattern; the tail loop is what makes the
            # bounded-transfer assertion non-vacuous — a padded full-capacity
            # transfer would deposit the sender's surplus there.
            for src in pl.range(nr):
                n_rows_i32 = pl.read(recv_counts, [src, 0])
                # Scalar read/write — a [1,1] INT32 tile.load fails ptoas
                # 32-byte row alignment (4 bytes).
                pl.write(recv_out, [src, 0], n_rows_i32)
                n_rows = pl.cast(n_rows_i32, pl.INDEX)
                base = src * mr
                for r in pl.range(n_rows):
                    flat_row = base + r
                    chunk = pl.load(result, [flat_row, 0], [1, SIZE])
                    pl.store(chunk, [flat_row, 0], out)
                for r in pl.range(n_rows, mr):
                    flat_row = base + r
                    chunk = pl.load(result, [flat_row, 0], [1, SIZE])
                    pl.store(chunk, [flat_row, 0], out)
            return out, recv_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[total, SIZE], pl.FP32],
            counts: pl.Tensor[[nr, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[total, SIZE], pl.FP32]],
            recv_out: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
            data: pl.InOut[pld.DistributedTensor[[total, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            recv_counts: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[total, SIZE], pl.FP32], pl.Tensor[[nr, 1], pl.INT32]]:
            """Chip orchestration: dispatch to exchange_step with bound windows."""
            return self.exchange_step(inp, counts, out, recv_out, data, signal, recv_counts)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, total, SIZE], pl.FP32],
            send_counts: pl.Tensor[[nr, nr, 1], pl.INT32],
            outputs: pl.Out[pl.Tensor[[nr, total, SIZE], pl.FP32]],
            recv_outputs: pl.Out[pl.Tensor[[nr, nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[nr, total, SIZE], pl.FP32], pl.Tensor[[nr, nr, 1], pl.INT32]]:
            """HOST orchestrator: allocate windows once, loop over ranks calling chip_orch."""
            data_buf = pld.alloc_window_buffer(total * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [total, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                recv = pld.window(recv_buf, [nr, 1], dtype=pl.INT32)
                self.chip_orch(
                    inputs[r],
                    send_counts[r],
                    outputs[r],
                    recv_outputs[r],
                    data,
                    sig,
                    recv,
                    device=r,
                )
            return outputs, recv_outputs

    return AllToAllVIntrinsicNRank


class TestL3TensorAllToAllVIntrinsic:
    """L3 distributed runtime: variable-size all-to-all via ``pld.tensor.all_to_all_v``."""

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_all_to_all_v_intrinsic(self, test_config, device_ids, n_ranks):
        """Compile and run variable-size all-to-all for P=2 or P=4.

        Each rank sends ``n_ranks - dest`` rows to each peer (variable counts).
        MAX_RECV=4 is the compile-time capacity, actual sends ≤ MAX_RECV.
        The test validates that the push-based decomposition produces the
        correct per-src per-dest exchange, and that ``recv_counts`` publishes
        the receive-side counts (no hardcoded ``n_rows = nr - rank`` on device).
        """
        if len(device_ids) < n_ranks:
            pytest.skip(f"all_to_all_v P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        nr = n_ranks
        mr = MAX_RECV
        total = nr * mr

        program = _build_all_to_all_v_program(nr, mr)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:nr],
                num_sub_workers=0,
            ),
        )

        # Build inputs: 3D host view [nr, total, SIZE] = per-rank flat 2D
        # Rank r sends to dest d: rows dest*mr+k for k=0..n_rows-1
        # Value = r*1000 + d*100 + k*10 + j%10
        # The TPUT transfers only [n_rows, SIZE] — rows beyond n_rows are not
        # pushed at all.  The receiver uses recv_counts to identify the valid
        # rows; the rest of its capacity slot is simply never written.
        inputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)
        send_counts = torch.zeros((nr, nr, 1), dtype=torch.int32)
        for r in range(nr):
            for d in range(nr):
                n_rows = nr - d  # variable send count
                send_counts[r, d, 0] = n_rows
                base = d * mr
                for k in range(n_rows):
                    for j in range(SIZE):
                        inputs[r, base + k, j] = float(r * 1000 + d * 100 + k * 10 + j % 10)

        outputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)
        recv_outputs = torch.zeros((nr, nr, 1), dtype=torch.int32)

        compiled(inputs, send_counts, outputs, recv_outputs)

        # Golden validation:
        # Rank rank receives from src the chunk that src sent to dest=rank.
        # Flat 2D layout: rows src*mr+k hold what src pushed for peer dest=rank.
        # recv_outputs[rank, src] must equal what src put in send_counts[src, rank].
        for rank in range(nr):
            for src in range(nr):
                n_rows = int(send_counts[src, rank, 0].item())
                assert int(recv_outputs[rank, src, 0].item()) == n_rows, (
                    f"P={nr} rank={rank} src={src}: recv_counts={int(recv_outputs[rank, src, 0].item())} "
                    f"!= expected send_counts={n_rows}"
                )
                base = src * mr
                for k in range(n_rows):
                    expected_row = inputs[src, rank * mr + k, :]
                    got_row = outputs[rank, base + k, :]
                    assert torch.allclose(got_row, expected_row, atol=1e-5), (
                        f"P={nr} rank={rank} src={src} row={k}: "
                        f"max diff = {(got_row - expected_row).abs().max().item()}"
                    )


def _effective_rows(count: int, max_recv: int) -> int:
    """Rows the kernel actually transfers and publishes: ``clamp(count, 0, MAX_RECV)``.

    Both rails apply the identical two-sided clamp, so this is the single golden
    for either lowering path.
    """
    return max(0, min(count, max_recv))


# Count matrices exercising the boundaries of the clamp. Each entry maps
# (n_ranks, max_recv) -> an [nr][nr] matrix of raw (unclamped) send counts.
_SKEW_CASES = {
    # Nothing to one peer, everything to another: the case the padded transfer
    # was worst at, and the one that exercises the rows == 0 push guard.
    "zero_and_full": lambda nr, mr: [[0 if d % 2 == 0 else mr for d in range(nr)] for _ in range(nr)],
    # Single row against full capacity — maximum skew with a non-empty push.
    "one_and_full": lambda nr, mr: [[1 if d % 2 == 0 else mr for d in range(nr)] for _ in range(nr)],
    # Above capacity: must clamp down to MAX_RECV, never push into the next
    # destination's slice of the peer window.
    "over_capacity": lambda nr, mr: [[mr + 3 for _ in range(nr)] for _ in range(nr)],
    # Negative counts: must floor at 0. Before the transfer extent depended on
    # this value a negative count was merely a strange TNOTIFY payload; it is now
    # a would-be negative transfer extent.
    "negative": lambda nr, mr: [[-2 if d % 2 == 0 else mr for d in range(nr)] for _ in range(nr)],
}


class TestL3TensorAllToAllVSkew:
    """Boundary coverage for the runtime-sized transfer: 0, 1, capacity, over, negative.

    Together with the identical cases in the HOST rail's
    ``test_l3_host_tensor_all_to_all_v.py`` and the managed CHIP rail's
    ``test_l2_tensor_all_to_all_v.py``, this is the wire-parity gate: all three
    lowering paths are driven with the same counts and checked against the same
    golden, so a divergence in any of them shows up as a golden mismatch.

    All three files are coverage-identical, and every ``(rank, src)`` pair is
    checked on ``recv_counts``, the payload rows and the surplus-row tail —
    including the self slot. The HOST file excluded its self slot once (#2546)
    because the rank's own staged input surfaced there even when nothing was
    transferred, but that was the undefined tail of the write-only ``out``
    buffer, not the window; every rail's consume loop now mirrors the whole
    window into ``out``.
    """

    @pytest.mark.parametrize("case", sorted(_SKEW_CASES))
    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_all_to_all_v_skewed_counts(self, test_config, device_ids, n_ranks, case):
        if len(device_ids) < n_ranks:
            pytest.skip(f"all_to_all_v P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        nr = n_ranks
        mr = MAX_RECV
        total = nr * mr
        raw = _SKEW_CASES[case](nr, mr)

        compiled = ir.compile(
            _build_all_to_all_v_program(nr, mr),
            platform=test_config.platform,
            distributed_config=DistributedConfig(device_ids=device_ids[:nr], num_sub_workers=0),
        )

        # Fill the FULL capacity slot of every destination, not just the rows
        # being sent, so an over-send would deposit recognisable data in the
        # padding rows and the assertion below would catch it.
        #
        # ``salt`` makes every (case, n_ranks) combination's payload unique.
        # Window memory is not zero-initialised AND persists across tests in the
        # same process, so without a salt an unwritten row can still hold the
        # identical pattern written by an earlier test — indistinguishable from
        # a real over-send, and a false failure.
        salt = (sorted(_SKEW_CASES).index(case) + 1) * 100000 + nr * 10000
        inputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)
        send_counts = torch.zeros((nr, nr, 1), dtype=torch.int32)
        for r in range(nr):
            for d in range(nr):
                send_counts[r, d, 0] = raw[r][d]
                base = d * mr
                for k in range(mr):
                    for j in range(SIZE):
                        inputs[r, base + k, j] = float(salt + r * 1000 + d * 100 + k * 10 + j % 10)

        outputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)
        recv_outputs = torch.zeros((nr, nr, 1), dtype=torch.int32)
        compiled(inputs, send_counts, outputs, recv_outputs)

        for rank in range(nr):
            for src in range(nr):
                n_rows = _effective_rows(raw[src][rank], mr)

                # recv_counts publishes the clamped count, never the raw one.
                got_count = int(recv_outputs[rank, src, 0].item())
                assert got_count == n_rows, (
                    f"P={nr} case={case} rank={rank} src={src}: recv_counts={got_count} "
                    f"!= clamped({raw[src][rank]}) = {n_rows}"
                )

                base = src * mr
                for k in range(n_rows):
                    expected_row = inputs[src, rank * mr + k, :]
                    got_row = outputs[rank, base + k, :]
                    assert torch.allclose(got_row, expected_row, atol=1e-5), (
                        f"P={nr} case={case} rank={rank} src={src} row={k}: "
                        f"max diff = {(got_row - expected_row).abs().max().item()}"
                    )

                # Rows past the transfer extent must not carry the sender's
                # surplus data — that is the observable signature of a bounded
                # transfer, since the padded version pushed exactly those rows.
                #
                # ``outputs`` mirrors the whole window: the consume loop copies
                # the valid rows and the tail rows into it, partitioned, so a
                # padded full-capacity transfer would deposit the sender's
                # surplus right here and be visible.
                #
                # Asserted as exactly zero, which is stronger than "not equal
                # to the surplus": the runtime zeroes a comm-domain window at
                # allocation, before the handle is published to peers
                # (``aclrtMemset`` in ``comm_hccl.cpp``'s ``alloc_domain``), so
                # a row no TPUT ever wrote must still read 0. Every payload here
                # is ``salt + ...`` with ``salt > 0``, so a transferred row can
                # never be mistaken for an untouched one.
                for k in range(n_rows, mr):
                    got_row = outputs[rank, base + k, :]
                    assert torch.all(got_row == 0.0), (
                        f"P={nr} case={case} rank={rank} src={src} row={k}: an untransferred row is "
                        f"not zero — the transfer is not bounded by the runtime count "
                        f"(got {got_row[:4].tolist()}...)"
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
