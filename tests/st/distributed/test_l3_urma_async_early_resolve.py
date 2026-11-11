# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: asynchronous URMA PUT with early dispatch (A17).

V1 completion is owned by the enclosing AIV task. ``allow_early_resolve`` lets
the scheduler pre-stage this task's consumers, but the consumer body must not
enter before the transfer task retires. The test uses the overwrite-after-
completion pattern: the local row is overwritten by a later task that consumes
the transfer destination, so an early body entry delivers the overwritten bytes
to the peers. The golden keeps v1 on the peer rows and v2 on the local row.

The program is class-based (``@pl.program``) on purpose: ``pl.submit`` accepts
``self.<kernel>`` and the JIT specializer's module-level-kernel surface does not.
The dep-gen artifacts must record ``early_dispatch: true`` for the transfer
task.

Hardware: A5 only; there is no simulator URMA backend.
"""

import json
import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig

SIZE = 4096
N_RANKS = 2


def _rank_inputs(n_ranks: int, *, round_index: int = 0) -> torch.Tensor:
    """Rank r fills only its own row: input[r, r, j] = base + r*1000 + j."""
    inputs = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
    base = round_index * 10000
    for r in range(n_ranks):
        inputs[r, r, :] = torch.arange(SIZE, dtype=torch.float32) + base + r * 1000
    return inputs


def _expected_exchange(n_ranks: int, *, round_index: int = 0) -> torch.Tensor:
    """Every rank ends with every row: out[rank, d, j] = base + d*1000 + j."""
    base = round_index * 10000
    d = torch.arange(n_ranks, dtype=torch.float32).view(1, -1, 1)
    j = torch.arange(SIZE, dtype=torch.float32).view(1, 1, -1)
    return (d * 1000 + j + base).expand(n_ranks, n_ranks, SIZE).contiguous()


def _expected_put(n_ranks: int, *, round_index: int = 0) -> torch.Tensor:
    """Peer rows keep v1; the local row was overwritten with v2 after completion."""
    expected = _expected_exchange(n_ranks, round_index=round_index)
    overwritten = _expected_exchange(n_ranks, round_index=round_index + 1)
    for r in range(n_ranks):
        expected[r, r, :] = overwritten[r, r, :]
    return expected


def _build_program():
    """A PUT exchange whose transfer dispatch opts into early resolution."""

    @pl.program
    class EarlyResolvePut:
        @pl.function(type=pl.FunctionType.InCore)
        def stage_put(
            self,
            inp: pl.Tensor[[N_RANKS, SIZE], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[N_RANKS, SIZE], pl.FP32]],
        ) -> pld.DistributedTensor[[N_RANKS, SIZE], pl.FP32]:
            ctx = pld.get_comm_ctx(data)
            my_rank = pld.rank(ctx)
            nranks = pld.nranks(ctx)
            row = pl.load(inp, [my_rank, 0], [1, SIZE])
            data = pl.store(row, [my_rank, 0], data)
            for peer in pl.range(nranks):
                if peer != my_rank:
                    pld.tensor.put_async(
                        data,
                        peer=peer,
                        src=data,
                        dst_offsets=[my_rank, 0],
                        src_offsets=[my_rank, 0],
                        shape=[1, SIZE],
                        engine=pld.CommEngine.URMA,
                    )
            return data

        @pl.function(type=pl.FunctionType.InCore)
        def overwrite_after_completion(
            self,
            inp2: pl.Tensor[[N_RANKS, SIZE], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[N_RANKS, SIZE], pl.FP32]],
        ) -> pld.DistributedTensor[[N_RANKS, SIZE], pl.FP32]:
            ctx = pld.get_comm_ctx(data)
            my_rank = pld.rank(ctx)
            row = pl.load(inp2, [my_rank, 0], [1, SIZE])
            return pl.store(row, [my_rank, 0], data)

        @pl.function(type=pl.FunctionType.InCore)
        def announce(
            self,
            data: pld.DistributedTensor[[N_RANKS, SIZE], pl.FP32],
            signal: pl.InOut[pld.DistributedTensor[[N_RANKS, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[N_RANKS, 1], pl.INT32]:
            ctx = pld.get_comm_ctx(signal)
            my_rank = pld.rank(ctx)
            nranks = pld.nranks(ctx)
            for peer in pl.range(nranks):
                if peer != my_rank:
                    pld.system.notify(
                        signal,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.Set,
                    )
            return signal

        @pl.function(type=pl.FunctionType.InCore)
        def await_peers(
            self,
            signal: pl.InOut[pld.DistributedTensor[[N_RANKS, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[N_RANKS, 1], pl.INT32]:
            ctx = pld.get_comm_ctx(signal)
            my_rank = pld.rank(ctx)
            nranks = pld.nranks(ctx)
            for peer in pl.range(nranks):
                if peer != my_rank:
                    pld.system.wait(
                        signal=signal,
                        offsets=[peer, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )
            return signal

        @pl.function(type=pl.FunctionType.InCore)
        def readback(
            self,
            data: pld.DistributedTensor[[N_RANKS, SIZE], pl.FP32],
            signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[N_RANKS, SIZE], pl.FP32]],
        ) -> pl.Tensor[[N_RANKS, SIZE], pl.FP32]:
            ctx = pld.get_comm_ctx(data)
            for row in pl.range(pld.nranks(ctx)):
                chunk = pl.load(data, [row, 0], [1, SIZE])
                out = pl.store(chunk, [row, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[N_RANKS, SIZE], pl.FP32],
            inp2: pl.Tensor[[N_RANKS, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[N_RANKS, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[N_RANKS, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[N_RANKS, 1], pl.INT32]],
        ) -> pl.Tensor[[N_RANKS, SIZE], pl.FP32]:
            # A17: the scheduler may pre-stage the consumers of this task; the
            # consumer body must still not enter before the transfer retires.
            staged, _ = pl.submit(self.stage_put, inp, data, allow_early_resolve=True)
            overwritten = self.overwrite_after_completion(inp2, staged)
            announced = self.announce(overwritten, signal)
            waited = self.await_peers(announced)
            return self.readback(overwritten, waited, out)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[N_RANKS, N_RANKS, SIZE], pl.FP32],
            inputs2: pl.Tensor[[N_RANKS, N_RANKS, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[N_RANKS, N_RANKS, SIZE], pl.FP32]],
        ) -> pl.Tensor[[N_RANKS, N_RANKS, SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(N_RANKS * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(N_RANKS * pl.INT32.get_byte())
            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [N_RANKS, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [N_RANKS, 1], dtype=pl.INT32)
                self.chip_orch(inputs[r], inputs2[r], outputs[r], data, sig, device=r)
            return outputs

    return EarlyResolvePut


@pytest.mark.platforms("a5")
class TestL3UrmaAsyncEarlyResolve:
    def test_early_resolve_does_not_release_early(self, test_config, device_ids):
        """A17: early dispatch must not let the consumer enter its body early."""
        n_ranks = N_RANKS
        if len(device_ids) < n_ranks:
            pytest.skip(f"URMA early-resolve needs {n_ranks} devices, got {device_ids}")

        compiled = ir.compile(
            _build_program(),
            platform=test_config.platform,
            distributed_config=DistributedConfig(device_ids=device_ids[:n_ranks], num_sub_workers=0),
        )
        inputs = _rank_inputs(n_ranks)
        inputs2 = _rank_inputs(n_ranks, round_index=1)
        outputs = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
        compiled(
            inputs,
            inputs2,
            outputs,
            config=RunConfig(platform=test_config.platform, enable_dep_gen=True),
        )

        expected = _expected_put(n_ranks)
        assert torch.allclose(outputs, expected), (
            f"early-resolve PUT mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

        dfx_base = compiled.output_dir / "dfx_outputs"
        for r in range(n_ranks):
            deps_path = dfx_base / f"rank{r}" / "d0" / "deps.json"
            assert deps_path.is_file(), f"missing deps.json: {deps_path}"
            deps = json.loads(deps_path.read_text())
            assert any(task.get("early_dispatch") is True for task in deps.get("tasks", [])), (
                f"rank {r}: no early_dispatch task in {deps_path}"
            )


if __name__ == "__main__":
    pytest.main([sys.argv[0], "-v", *sys.argv[1:]])
