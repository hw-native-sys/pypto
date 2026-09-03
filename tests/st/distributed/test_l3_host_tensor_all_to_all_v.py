# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST: host-orchestrator ``pld.tensor.all_to_all_v`` builtin dispatch.

Validates the HOST-level variable-size all-to-all (MPI_Alltoallv pattern) lowers
through ``LowerHostTensorCollectives`` and produces correct rank-ordered
personalized exchange via the hand-written ``builtin.tensor.all_to_all_v``
kernel — the HOST-orchestrated counterpart of the InCore-only
``test_l3_tensor_all_to_all_v_intrinsic.py``.

The HOST lowering path detects ``pld.tensor.all_to_all_v`` in ``host_orch`` and
lowers it to ``builtin.tensor.all_to_all_v`` per chip. The exchange uses a
push-based TPUT pattern with FIVE window-bound resources:

  1. **Stage** (``stage_step``): each rank writes its per-destination chunks
     into ``input_buf`` — a window used ONLY as a TPUT source, never as an
     incoming-push destination (same discipline as symmetric all_to_all's HOST
     builtin).
  2. **Fill counts** (``fill_counts_step``): each rank writes its own
     per-destination send counts into ``counts_buf``. This staging step only
     exists because the HOST builtin narrows ``send_counts`` from the InCore
     composite's ``Tensor``-or-``DistributedTensor`` contract down to a strict
     window-bound ``DistributedTensor`` — ``EmitBuiltinWindowCollectiveDispatch``
     has no dispatch path for a plain ``Tensor`` arg. A real ergonomic cost of
     the narrowing, not a test artifact.
  3. **All-to-all-v** (``builtin.tensor.all_to_all_v``): the kernel pushes only
     ``rows = clamp(send_counts[dest], 0, MAX_RECV)`` rows per destination into
     ``data_buf`` — the padding up to ``MAX_RECV`` never crosses the wire —
     publishes that same clamped count into peer ``recv_counts[my_rank, 0]``
     via TNOTIFY, and synchronises with one barrier. The clamp is two-sided and
     identical to ``LowerTensorAllToAllVRule``'s, keeping the HOST and InCore
     rails bit-for-bit identical on the wire for every input.
  4. **Consume** (``consume_step``): each rank reads ``recv_counts`` to learn
     how many rows each source actually sent, then reads back only those valid
     rows from ``data_buf``.

``input_buf``, ``data_buf``, ``signal_buf``, ``counts_buf``, and ``recv_buf``
must all be allocated inside the SAME host_orch comm-domain scope —
``EmitBuiltinWindowCollectiveDispatch`` requires every window-bound arg of one
builtin call to share a comm-domain handle.

The HOST kernel derives ``MAX_RECV`` at entry from the runtime rank count
(``target.shape[0] / nranks``), so the per-destination block size is always
consistent with the devices actually running. The program is still built via
a factory function per parametrized rank count (signal/counts shapes are
per-rank-count), matching ``test_l3_tensor_all_to_all_v_intrinsic.py``'s
pattern.

ST coverage: P=2 and P=4 (skips when fewer devices are available).
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


def _build_host_all_to_all_v_program(n_ranks: int, max_recv: int):
    """Build an N-rank HOST-orchestrated variable-size all-to-all program."""
    nr = n_ranks
    mr = max_recv
    total = nr * mr

    @pl.program
    class HostTensorAllToAllV:
        """N-rank HOST-orchestrated variable-size all-to-all program."""

        @pl.function(type=pl.FunctionType.InCore)
        def stage_step(
            self,
            inp: pl.Tensor[[total, SIZE], pl.FP32],
            stage: pl.Out[pld.DistributedTensor[[total, SIZE], pl.FP32]],
        ):
            for row in pl.range(total):
                chunk = pl.load(inp, [row, 0], [1, SIZE])
                stage = pl.store(chunk, [row, 0], stage)

        @pl.function(type=pl.FunctionType.Orchestration)
        def stage_orch(
            self,
            inp: pl.Tensor[[total, SIZE], pl.FP32],
            stage: pl.Out[pld.DistributedTensor[[total, SIZE], pl.FP32]],
        ):
            self.stage_step(inp, stage)

        @pl.function(type=pl.FunctionType.InCore)
        def fill_counts_step(
            self,
            counts_row: pl.Tensor[[nr, 1], pl.INT32],
            counts: pl.Out[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ):
            # Scalar read/write — a [1,1] INT32 tile.load/store fails ptoas
            # 32-byte row alignment (4 bytes); same pitfall the InCore
            # all_to_all_v intrinsic ST avoids for recv_counts.
            for d in pl.range(nr):
                v = pl.read(counts_row, [d, 0])
                pl.write(counts, [d, 0], v)

        @pl.function(type=pl.FunctionType.Orchestration)
        def fill_counts_orch(
            self,
            counts_row: pl.Tensor[[nr, 1], pl.INT32],
            counts: pl.Out[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ):
            self.fill_counts_step(counts_row, counts)

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            data: pld.DistributedTensor[[total, SIZE], pl.FP32],
            recv_counts: pld.DistributedTensor[[nr, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[total, SIZE], pl.FP32]],
            recv_out: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[total, SIZE], pl.FP32], pl.Tensor[[nr, 1], pl.INT32]]:
            # Two loops whose row ranges partition each sender's slot, so every
            # row of the window reaches `out` exactly once and `out` ends up
            # FULLY written. That matters: a `pl.Out` tensor is write-only on
            # the device (the host buffer is never uploaded), so a row the
            # kernel skipped would come back as undefined memory rather than as
            # window content — and the host-side tail check would be inspecting
            # that, not the window.
            #   [base, base + recv_counts[src])       valid -- checked vs golden
            #   [base + recv_counts[src], base + mr)  tail  -- bounded-transfer check
            for src in pl.range(nr):
                n_rows_i32 = pl.read(recv_counts, [src, 0])
                pl.write(recv_out, [src, 0], n_rows_i32)
                n_rows = pl.cast(n_rows_i32, pl.INDEX)
                base = src * mr
                for r in pl.range(n_rows):
                    flat_row = base + r
                    chunk = pl.load(data, [flat_row, 0], [1, SIZE])
                    out = pl.store(chunk, [flat_row, 0], out)
                for r in pl.range(n_rows, mr):
                    flat_row = base + r
                    chunk = pl.load(data, [flat_row, 0], [1, SIZE])
                    out = pl.store(chunk, [flat_row, 0], out)
            return out, recv_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def consume_orch(
            self,
            data: pld.DistributedTensor[[total, SIZE], pl.FP32],
            recv_counts: pld.DistributedTensor[[nr, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[total, SIZE], pl.FP32]],
            recv_out: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[total, SIZE], pl.FP32], pl.Tensor[[nr, 1], pl.INT32]]:
            return self.consume_step(data, recv_counts, out, recv_out)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, total, SIZE], pl.FP32],
            send_counts: pl.Tensor[[nr, nr, 1], pl.INT32],
            outputs: pl.Out[pl.Tensor[[nr, total, SIZE], pl.FP32]],
            recv_outputs: pl.Out[pl.Tensor[[nr, nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[nr, total, SIZE], pl.FP32], pl.Tensor[[nr, nr, 1], pl.INT32]]:
            input_buf = pld.alloc_window_buffer(total * SIZE * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(total * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())
            counts_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                stage = pld.window(input_buf, [total, SIZE], dtype=pl.FP32)
                self.stage_orch(inputs[r], stage, device=r)

            for r in pl.range(pld.world_size()):
                counts = pld.window(counts_buf, [nr, 1], dtype=pl.INT32)
                self.fill_counts_orch(send_counts[r], counts, device=r)

            stage = pld.window(input_buf, [total, SIZE], dtype=pl.FP32)
            data = pld.window(data_buf, [total, SIZE], dtype=pl.FP32)
            signal = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
            counts = pld.window(counts_buf, [nr, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [nr, 1], dtype=pl.INT32)
            data = pld.tensor.all_to_all_v(stage, data, signal, counts, recv)

            for r in pl.range(pld.world_size()):
                self.consume_orch(data, recv, outputs[r], recv_outputs[r], device=r)

            return outputs, recv_outputs

    return HostTensorAllToAllV


class TestL3HostTensorAllToAllV:
    """L3 distributed runtime: HOST-level variable-size all-to-all via builtin dispatch."""

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_host_tensor_all_to_all_v(self, test_config, device_ids, n_ranks):
        """Compile and run host-level all_to_all_v for P in {2, 4}.

        Each rank sends ``n_ranks - dest`` rows to each destination (variable,
        runtime-dependent counts) — same golden pattern as the InCore-only ST
        (``test_l3_tensor_all_to_all_v_intrinsic.py``).
        """
        if len(device_ids) < n_ranks:
            pytest.skip(f"host all_to_all_v P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        nr = n_ranks
        mr = MAX_RECV
        total = nr * mr

        program = _build_host_all_to_all_v_program(nr, mr)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:nr],
                num_sub_workers=0,
            ),
        )

        variant_dir = compiled.output_dir / "next_levels" / "builtin.tensor.all_to_all_v__fp32"
        assert variant_dir.is_dir(), f"expected {variant_dir}"
        assert (variant_dir / "kernel_config.py").is_file()

        # Rank r sends to dest d: rows dest*mr+k for k=0..n_rows-1.
        # Value = r*1000 + d*100 + k*10 + j%10 (same formula as the InCore ST).
        # The TPUT transfers only [n_rows, SIZE] per destination — rows beyond
        # n_rows are never pushed. The receiver uses recv_counts to identify the
        # valid rows; the rest of its capacity slot stays unwritten.
        inputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)
        send_counts = torch.zeros((nr, nr, 1), dtype=torch.int32)
        for r in range(nr):
            for d in range(nr):
                n_rows = nr - d  # variable send count, same golden formula as the InCore ST
                send_counts[r, d, 0] = n_rows
                base = d * mr
                for k in range(n_rows):
                    for j in range(SIZE):
                        inputs[r, base + k, j] = float(r * 1000 + d * 100 + k * 10 + j % 10)

        outputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)
        recv_outputs = torch.zeros((nr, nr, 1), dtype=torch.int32)

        compiled(inputs, send_counts, outputs, recv_outputs)

        # Rank rank receives from src the chunk that src sent to dest=rank.
        for rank in range(nr):
            for src in range(nr):
                n_rows = int(send_counts[src, rank, 0].item())
                assert int(recv_outputs[rank, src, 0].item()) == n_rows, (
                    f"P={nr} rank={rank} src={src}: recv_counts="
                    f"{int(recv_outputs[rank, src, 0].item())} != expected send_counts={n_rows}"
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
    """Rows actually transferred and published: ``clamp(count, 0, MAX_RECV)``."""
    return max(0, min(count, max_recv))


# Deliberately duplicated from the InCore ST
# (``collectives/test_l3_tensor_all_to_all_v_intrinsic.py``) rather than
# imported: the point is to drive BOTH lowering rails independently with the
# same counts against the same golden. If the rails ever diverge on the wire,
# one of the two files fails. Keep the two case tables in sync.
_SKEW_CASES = {
    "zero_and_full": lambda nr, mr: [[0 if d % 2 == 0 else mr for d in range(nr)] for _ in range(nr)],
    "one_and_full": lambda nr, mr: [[1 if d % 2 == 0 else mr for d in range(nr)] for _ in range(nr)],
    "over_capacity": lambda nr, mr: [[mr + 3 for _ in range(nr)] for _ in range(nr)],
    "negative": lambda nr, mr: [[-2 if d % 2 == 0 else mr for d in range(nr)] for _ in range(nr)],
}


class TestL3HostTensorAllToAllVSkew:
    """HOST-rail boundary coverage: 0, 1, capacity, over-capacity, negative counts.

    Mirrors the InCore cases in
    ``collectives/test_l3_tensor_all_to_all_v_intrinsic.py`` — same counts, same
    golden — so a wire divergence between the rails surfaces as a golden
    mismatch in one file or the other.

    Every ``(rank, src)`` pair is checked on all three counts — ``recv_counts``,
    the payload rows, and the surplus-row tail — including the self slot. The
    self slot was excluded once (#2546) because the rank's own staged input
    surfaced there even when nothing was transferred, but that was the undefined
    tail of the write-only ``out`` buffer, not the window; the consume loop now
    mirrors the whole window into ``out``, so the tail check is meaningful on
    every pair.
    """

    @pytest.mark.parametrize("case", sorted(_SKEW_CASES))
    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_host_all_to_all_v_skewed_counts(self, test_config, device_ids, n_ranks, case):
        if len(device_ids) < n_ranks:
            pytest.skip(f"host all_to_all_v P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        nr = n_ranks
        mr = MAX_RECV
        total = nr * mr
        raw = _SKEW_CASES[case](nr, mr)

        compiled = ir.compile(
            _build_host_all_to_all_v_program(nr, mr),
            platform=test_config.platform,
            distributed_config=DistributedConfig(device_ids=device_ids[:nr], num_sub_workers=0),
        )

        # Fill every destination's FULL capacity slot so an over-send would be
        # visible as written padding rows below. ``salt`` keeps each
        # (case, n_ranks) payload unique: window memory is not zero-initialised
        # and persists across tests in the same process, so an unwritten row can
        # otherwise hold the identical pattern an earlier test wrote — which is
        # indistinguishable from a real over-send and fails spuriously.
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

                # The sender's surplus rows must never arrive. `outputs` now
                # mirrors the whole window (the consume loop copies the tail
                # rows too), so this inspects window content rather than an
                # unwritten region of a write-only `pl.Out` buffer.
                #
                # Asserted as exactly zero: the runtime zeroes a comm-domain
                # window at allocation, before the handle is published to peers
                # (``aclrtMemset`` in ``comm_hccl.cpp``'s ``alloc_domain``), so
                # a row no TPUT ever wrote must still read 0. Every payload is
                # ``salt + ...`` with ``salt > 0``, so a transferred row can
                # never be mistaken for an untouched one.
                #
                # The self slot is no longer skipped. It was excluded because
                # the rank's own staged input showed up here even when nothing
                # was transferred (#2546) — but that was the undefined `out`
                # tail, not the window.
                for k in range(n_rows, mr):
                    got_row = outputs[rank, base + k, :]
                    assert torch.all(got_row == 0.0), (
                        f"P={nr} case={case} rank={rank} src={src} row={k}: an untransferred row is "
                        f"not zero — the HOST transfer is not bounded by the runtime count "
                        f"(got {got_row[:4].tolist()}...)"
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
