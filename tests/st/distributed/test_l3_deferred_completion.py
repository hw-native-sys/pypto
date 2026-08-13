# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST for deferred task completion and task-start freshness.

On A2/A3, each of two ranks fills every available AIV with a wait task before
submitting the task that publishes to its peer. The simulator exposes 16 AIVs
(``SIM_AUTO_BLOCKDIM=8``, two AIVs per block); full A2/A3 hardware exposes 48:

* every waiter registers ``signal[0, 0] >= 1`` with
  :func:`pld.system.defer_wait` and returns its physical core;
* each consumer is gated by ordinary ``deps=[wait_tid]``; Simpler's standard
  executor invalidates the data cache at every task start before reading payload;
* only after all waiter/consumer pairs have been submitted does a publisher
  write a rank-specific payload into the peer's window and notify its signal.

Replacing the deferred wait with a blocking ``pld.system.wait`` in this
saturation topology occupies every AIV core before the publisher can run and
deadlocks. Completion therefore proves physical-core release as well as
cross-rank liveness. Every consumer must read the peer's tag, which also checks
that no consumer runs before counter readiness and that Simpler's standard
task-start cache invalidation exposes the payload.

This core count is intentionally distinct from Simpler's
``MAX_ASYNC_WAITS=64`` scheduler-list capacity and its
``MAX_COMPLETIONS_PER_TASK=64`` condition cap. The normal liveness test stays
at the platform's AIV saturation point; it does not treat either unrelated
runtime capacity as a core-release requirement.

The persistent case runs the same compiled graph for monotonically increasing
epochs with persistent-window reset explicitly disabled. The first payload
lane and the signal both carry the epoch, so a consumer that observes stale
payload or a waiter registration left behind by the prior call fails. A
separate one-waiter case holds the epoch at one after priming the signal, which
exercises registration and wait-table reuse when the counter is already ready.

A smaller four-waiter A5 case checks the same cross-rank payload, completion
dependency, and persistent-reset contract without making an A5 core-saturation
claim.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

_N_RANKS = 2
_A2A3_SIM_AIVS = 16
_A2A3_ONBOARD_AIVS = 48
_A5_SMOKE_WAITS = 4
_PAYLOAD_WIDTH = 8  # 1x8 INT32 is the minimum 32-byte vector tile.


def _build_deferred_completion_program(waiter_count: int):
    """Build the deferred-wait graph for ``waiter_count`` registrations/rank."""

    WAITER_COUNT = waiter_count  # noqa: N806 - closed over by IR shape annotations

    @pl.program
    class DeferredCompletion:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, _PAYLOAD_WIDTH], pl.INT32],
            out: pl.Out[pl.Tensor[[WAITER_COUNT, _PAYLOAD_WIDTH], pl.INT32]],
            payload: pl.InOut[pld.DistributedTensor[[1, _PAYLOAD_WIDTH], pl.INT32]],
            signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[WAITER_COUNT, _PAYLOAD_WIDTH], pl.INT32]:
            # Manual mode is load-bearing: the only waiter -> consumer edge is
            # the explicit TaskId edge, while the later publisher must remain
            # independent so released physical cores can make forward progress.
            with pl.manual_scope():
                for slot in pl.range(WAITER_COUNT):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="deferred_wait") as wait_tid:
                        epoch: pl.Scalar[pl.INT32] = pl.read(inp, [0, 0])
                        pld.system.defer_wait(
                            signal=signal,
                            offsets=[0, 0],
                            expected=epoch,
                            cmp=pld.WaitCmp.Ge,
                        )

                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="completion_consumer",
                        deps=[wait_tid],
                    ) as _consumer_tid:
                        received = pl.load(payload, [0, 0], [1, _PAYLOAD_WIDTH])
                        out = pl.store(received, [slot, 0], out)

                # Submitted last on purpose. A blocking version at
                # the platform's AIV saturation count cannot reach this task
                # after all AIV cores enter TWAIT; deferred waiters return
                # their cores immediately.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="delayed_publisher"):
                    local_payload = pl.load(inp, [0, 0], [1, _PAYLOAD_WIDTH])
                    pld.tile.remote_store(local_payload, target=payload, peer=peer, offsets=[0, 0])
                    epoch: pl.Scalar[pl.INT32] = pl.read(inp, [0, 0])
                    pld.system.notify(
                        target=signal,
                        peer=peer,
                        offsets=[0, 0],
                        value=epoch,
                        op=pld.NotifyOp.Set,
                    )
            return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[_N_RANKS, 1, _PAYLOAD_WIDTH], pl.INT32],
            outputs: pl.Out[pl.Tensor[[_N_RANKS, WAITER_COUNT, _PAYLOAD_WIDTH], pl.INT32]],
        ) -> pl.Tensor[[_N_RANKS, WAITER_COUNT, _PAYLOAD_WIDTH], pl.INT32]:
            payload_buf = pld.alloc_window_buffer(_PAYLOAD_WIDTH * pl.INT32.get_byte())
            signal_buf = pld.alloc_window_buffer(pl.INT32.get_byte())

            for rank in pl.range(pld.world_size()):
                payload = pld.window(payload_buf, [1, _PAYLOAD_WIDTH], dtype=pl.INT32)
                signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                self.chip_orch(
                    inputs[rank],
                    outputs[rank],
                    payload,
                    signal,
                    (rank + 1) % pld.world_size(),
                    device=rank,
                )
            return outputs

    return DeferredCompletion


def _a2a3_saturation_waiters(platform: str) -> int:
    """Return the AIV saturation point for the selected A2/A3 target."""
    return _A2A3_SIM_AIVS if str(platform).endswith("sim") else _A2A3_ONBOARD_AIVS


def _compile_two_rank(program, test_config, device_ids, output_dir):
    """Compile one deferred-completion variant for exactly two ranks."""
    if len(device_ids) < _N_RANKS:
        pytest.skip(f"deferred completion needs {_N_RANKS} devices, got {device_ids}")
    return ir.compile(
        program,
        output_dir=str(output_dir),
        platform=test_config.platform,
        distributed_config=DistributedConfig(
            device_ids=device_ids[:_N_RANKS],
            num_sub_workers=0,
        ),
    )


def _expected(waiter_count: int, epoch: int) -> torch.Tensor:
    """Return the current epoch followed by the peer's rank tag."""
    expected = torch.empty((_N_RANKS, waiter_count, _PAYLOAD_WIDTH), dtype=torch.int32)
    expected[0].fill_(2)
    expected[1].fill_(1)
    expected[:, :, 0].fill_(epoch)
    return expected


def _inputs(epoch: int = 1) -> torch.Tensor:
    """Return a shared epoch lane followed by rank-specific payload lanes."""
    inputs = torch.empty((_N_RANKS, 1, _PAYLOAD_WIDTH), dtype=torch.int32)
    inputs[0].fill_(1)
    inputs[1].fill_(2)
    inputs[:, :, 0].fill_(epoch)
    return inputs.share_memory_()


def _assert_peer_tags(outputs: torch.Tensor, waiter_count: int, epoch: int = 1) -> None:
    expected = _expected(waiter_count, epoch)
    assert torch.equal(outputs, expected), (
        "deferred-completion payload mismatch: "
        f"got rank0={outputs[0].flatten().tolist()}, rank1={outputs[1].flatten().tolist()}"
    )


@pytest.mark.platforms("a2a3", "a2a3sim")
def test_deferred_completion_releases_cores_and_reuses_wait_table(test_config, device_ids, tmp_path):
    """An AIV-saturating wait set releases cores across monotonic epochs."""
    waiter_count = _a2a3_saturation_waiters(test_config.platform)
    program = _build_deferred_completion_program(waiter_count)
    compiled = _compile_two_rank(program, test_config, device_ids, tmp_path / "liveness")
    outputs = torch.full(
        (_N_RANKS, waiter_count, _PAYLOAD_WIDTH),
        -1,
        dtype=torch.int32,
    ).share_memory_()
    # Prepared chip workers are forked by ``prepare``.  Create every shared
    # host tensor first, then update the retained mapping in place per epoch.
    inputs = _inputs()

    with compiled.prepare(
        config=test_config,
        persistent=True,
        reset_persistent_windows=False,
    ) as worker:
        for epoch in range(1, 4):
            inputs[0].fill_(1)
            inputs[1].fill_(2)
            inputs[:, :, 0].fill_(epoch)
            outputs.fill_(-1)
            worker(inputs, outputs, config=test_config)
            _assert_peer_tags(outputs, waiter_count, epoch)


@pytest.mark.platforms("a2a3", "a2a3sim")
def test_deferred_completion_reuses_already_ready_registration(test_config, device_ids, tmp_path):
    """A primed counter completes later registrations and reuses wait slots."""
    program = _build_deferred_completion_program(1)
    compiled = _compile_two_rank(program, test_config, device_ids, tmp_path / "already_ready")
    inputs = _inputs(epoch=1)
    outputs = torch.full((_N_RANKS, 1, _PAYLOAD_WIDTH), -1, dtype=torch.int32).share_memory_()

    with compiled.prepare(
        config=test_config,
        persistent=True,
        reset_persistent_windows=False,
    ) as worker:
        # The first call primes signal=1 after registering against a fresh
        # counter. The next two calls register against that already-ready
        # value and exercise immediate completion plus wait-slot reuse. The
        # repeated payload is intentional: freshness across generations is
        # covered by the monotonic-epoch saturation case above.
        for _ in range(3):
            outputs.fill_(-1)
            worker(inputs, outputs, config=test_config)
            _assert_peer_tags(outputs, 1, epoch=1)


@pytest.mark.platforms("a5", "a5sim")
def test_deferred_completion_a5_cross_rank_correctness(test_config, device_ids, tmp_path):
    """A small A5 case preserves correctness over two monotonic epochs without a window reset."""
    program = _build_deferred_completion_program(_A5_SMOKE_WAITS)
    compiled = _compile_two_rank(program, test_config, device_ids, tmp_path / "a5_correctness")
    outputs = torch.full(
        (_N_RANKS, _A5_SMOKE_WAITS, _PAYLOAD_WIDTH),
        -1,
        dtype=torch.int32,
    ).share_memory_()
    inputs = _inputs()

    with compiled.prepare(
        config=test_config,
        persistent=True,
        reset_persistent_windows=False,
    ) as worker:
        for epoch in range(1, 3):
            inputs[0].fill_(1)
            inputs[1].fill_(2)
            inputs[:, :, 0].fill_(epoch)
            outputs.fill_(-1)
            worker(inputs, outputs, config=test_config)
            _assert_peer_tags(outputs, _A5_SMOKE_WAITS, epoch)


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
