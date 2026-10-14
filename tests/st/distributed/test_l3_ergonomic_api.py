# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST: ergonomic ``pld.*`` collective wrappers (plan 58).

Each test drives the HOST auto-signal wrapper (``pld.all_reduce`` /
``pld.broadcast`` / ``pld.all_gather`` / ``pld.barrier``) through the real
compiler and run path, with per-rank inputs so the goldens validate the
cross-rank exchange (not just local pass-through). The underlying collective
semantics are additionally covered by the ``test_l3_host_tensor_*.py`` siblings;
here the point is the auto-managed signal + parser resolution end-to-end.

``all_to_all_v`` is intentionally not run end-to-end here: the HOST
``builtin.tensor.all_to_all_v`` rail is plan 65 / #2243 (not on ``main`` yet).
Its wrapper delegation is unit-tested in ``tests/ut/language/test_collective_api.py``.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64
NR = 2


def _make_rank_inputs(n_ranks: int, size: int = SIZE) -> torch.Tensor:
    """Per-rank distinct inputs: rank r is [r*100, r*100+size)."""
    rows = [
        torch.arange(r * 100.0, r * 100.0 + size, dtype=torch.float32).reshape(1, size)
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


def _compile(program, test_config, device_ids):
    return ir.compile(
        program,
        platform=test_config.platform,
        distributed_config=DistributedConfig(
            device_ids=device_ids[:NR],
            num_sub_workers=0,
        ),
    )


def _assert_close(outputs, expected, label):
    assert torch.allclose(outputs, expected), (
        f"{label} mismatch: max diff = {(outputs - expected).abs().max().item()}"
    )


@pl.program
class ErgonomicAllReduce:
    @pl.function(type=pl.FunctionType.InCore)
    def publish_step(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
        return pl.store(pl.load(inp, [0, 0], [1, SIZE]), [0, 0], data)

    @pl.function(type=pl.FunctionType.Orchestration)
    def publish_orch(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
        return self.publish_step(inp, data)

    @pl.function(type=pl.FunctionType.InCore)
    def consume_step(
        self,
        data: pld.DistributedTensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        return pl.store(pl.load(data, [0, 0], [1, SIZE]), [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def consume_orch(
        self,
        data: pld.DistributedTensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        return self.consume_step(data, out)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[NR, 1, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[NR, 1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, 1, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
            self.publish_orch(inputs[r], data, device=r)
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        data = pld.all_reduce(data, op=pld.ReduceOp.Sum)  # mesh: signal auto-synthesized
        for r in pl.range(pld.world_size()):
            self.consume_orch(data, outputs[r], device=r)
        return outputs


@pl.program
class ErgonomicAllReduceRing:
    @pl.function(type=pl.FunctionType.InCore)
    def publish_step(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
        return pl.store(pl.load(inp, [0, 0], [1, SIZE]), [0, 0], data)

    @pl.function(type=pl.FunctionType.Orchestration)
    def publish_orch(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
        return self.publish_step(inp, data)

    @pl.function(type=pl.FunctionType.InCore)
    def consume_step(
        self,
        data: pld.DistributedTensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        return pl.store(pl.load(data, [0, 0], [1, SIZE]), [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def consume_orch(
        self,
        data: pld.DistributedTensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        return self.consume_step(data, out)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[NR, 1, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[NR, 1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, 1, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
        for r in pl.range(NR):
            data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
            self.publish_orch(inputs[r], data, device=r)
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        data = pld.all_reduce(data, op=pld.ReduceOp.Sum, mode="ring", nranks=NR)
        for r in pl.range(NR):
            self.consume_orch(data, outputs[r], device=r)
        return outputs


@pl.program
class ErgonomicBroadcast:
    @pl.function(type=pl.FunctionType.InCore)
    def publish_step(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
        if my_rank == 0:  # root stages only
            local = pl.load(inp, [0, 0], [1, SIZE])
            return pl.store(local, [0, 0], data)
        return data

    @pl.function(type=pl.FunctionType.Orchestration)
    def publish_orch(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        my_rank: pl.Scalar[pl.INT32],
    ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
        return self.publish_step(inp, data, my_rank)

    @pl.function(type=pl.FunctionType.InCore)
    def consume_step(
        self,
        data: pld.DistributedTensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        return pl.store(pl.load(data, [0, 0], [1, SIZE]), [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def consume_orch(
        self,
        data: pld.DistributedTensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        return self.consume_step(data, out)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[NR, 1, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[NR, 1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, 1, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
            self.publish_orch(inputs[r], data, r, device=r)
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        data = pld.broadcast(data, root=0)
        for r in pl.range(NR):
            self.consume_orch(data, outputs[r], device=r)
        return outputs


@pl.program
class ErgonomicAllGather:
    @pl.function(type=pl.FunctionType.InCore)
    def publish_step(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        local: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
        return pl.store(pl.load(inp, [0, 0], [1, SIZE]), [0, 0], local)

    @pl.function(type=pl.FunctionType.Orchestration)
    def publish_orch(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        local: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
        return self.publish_step(inp, local)

    @pl.function(type=pl.FunctionType.InCore)
    def consume_step(
        self,
        target: pld.DistributedTensor[[NR, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[NR, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, SIZE], pl.FP32]:
        # Copy the whole gathered target so every rank's chunk is validated.
        return pl.store(pl.load(target, [0, 0], [NR, SIZE]), [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def consume_orch(
        self,
        target: pld.DistributedTensor[[NR, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[NR, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, SIZE], pl.FP32]:
        return self.consume_step(target, out)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[NR, 1, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[NR, NR, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, NR, SIZE], pl.FP32]:
        local_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
        target_buf = pld.alloc_window_buffer(NR * SIZE * pl.FP32.get_byte())
        for r in pl.range(NR):
            local = pld.window(local_buf, [1, SIZE], dtype=pl.FP32)
            self.publish_orch(inputs[r], local, device=r)
        local = pld.window(local_buf, [1, SIZE], dtype=pl.FP32)
        target = pld.window(target_buf, [NR, SIZE], dtype=pl.FP32)
        target = pld.all_gather(local, target)
        for r in pl.range(NR):
            self.consume_orch(target, outputs[r], device=r)
        return outputs


@pl.program
class ErgonomicBarrier:
    @pl.function(type=pl.FunctionType.InCore)
    def publish_step(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
        return pl.store(pl.load(inp, [0, 0], [1, SIZE]), [0, 0], data)

    @pl.function(type=pl.FunctionType.Orchestration)
    def publish_orch(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        sig: pld.DistributedTensor[[NR], pl.INT32],
    ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
        # ``sig`` is declared so the dispatch tags the signal window with
        # comm-domain coverage (a barrier-only signal cannot be auto-covered
        # on ``main`` — see ``pld.barrier``'s docstring).
        return self.publish_step(inp, data)

    @pl.function(type=pl.FunctionType.InCore)
    def consume_step(
        self,
        data: pld.DistributedTensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        peer: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        recv = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, SIZE])
        return pl.store(recv, [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def consume_orch(
        self,
        data: pld.DistributedTensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        peer: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        return self.consume_step(data, out, peer)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[NR, 1, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[NR, 1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, 1, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
        signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())
        signal = pld.window(signal_buf, [pld.world_size()], dtype=pl.INT32)

        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
            self.publish_orch(inputs[r], data, signal, device=r)

        signal = pld.barrier(signal)  # covered signal; all ranks must reach it

        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
            peer = (r + 1) % pld.world_size()
            self.consume_orch(data, outputs[r], peer, device=r)
        return outputs


class TestErgonomicCollectiveApi:
    def test_all_reduce_mesh(self, test_config, device_ids):
        if len(device_ids) < NR:
            pytest.skip(f"ergonomic allreduce P={NR} needs {NR} devices, got {device_ids}")
        compiled = _compile(ErgonomicAllReduce, test_config, device_ids)
        inputs = _make_rank_inputs(NR)
        outputs = torch.zeros_like(inputs)
        compiled(inputs, outputs)
        expected = torch.stack([inputs.sum(dim=0)] * NR)
        _assert_close(outputs, expected, "mesh AR")

    def test_all_reduce_ring(self, test_config, device_ids):
        if len(device_ids) < NR:
            pytest.skip(f"ergonomic ring allreduce P={NR} needs {NR} devices, got {device_ids}")
        compiled = _compile(ErgonomicAllReduceRing, test_config, device_ids)
        inputs = _make_rank_inputs(NR)
        outputs = torch.zeros_like(inputs)
        compiled(inputs, outputs)
        expected = torch.stack([inputs.sum(dim=0)] * NR)
        _assert_close(outputs, expected, "ring AR")

    def test_broadcast(self, test_config, device_ids):
        if len(device_ids) < NR:
            pytest.skip(f"ergonomic broadcast P={NR} needs {NR} devices, got {device_ids}")
        compiled = _compile(ErgonomicBroadcast, test_config, device_ids)
        inputs = _make_rank_inputs(NR)
        outputs = torch.zeros_like(inputs)
        compiled(inputs, outputs)
        expected = torch.stack([inputs[0]] * NR)
        _assert_close(outputs, expected, "broadcast")

    def test_all_gather(self, test_config, device_ids):
        if len(device_ids) < NR:
            pytest.skip(f"ergonomic all_gather P={NR} needs {NR} devices, got {device_ids}")
        compiled = _compile(ErgonomicAllGather, test_config, device_ids)
        inputs = _make_rank_inputs(NR)
        outputs = torch.zeros((NR, NR, SIZE), dtype=inputs.dtype, device=inputs.device)
        compiled(inputs, outputs)
        # Every rank reads the full gathered target: output[r] holds all rank chunks.
        expected = torch.stack([inputs[:, 0, :]] * NR)
        _assert_close(outputs, expected, "all_gather")

    def test_barrier(self, test_config, device_ids):
        if len(device_ids) < NR:
            pytest.skip(f"ergonomic barrier P={NR} needs {NR} devices, got {device_ids}")
        compiled = _compile(ErgonomicBarrier, test_config, device_ids)
        inputs = _make_rank_inputs(NR)
        outputs = torch.zeros_like(inputs)
        compiled(inputs, outputs)
        # Each rank remote-loads its peer's published data after the barrier —
        # validates the barrier actually synchronized the ranks.
        expected = torch.stack([inputs[(r + 1) % NR] for r in range(NR)])
        assert torch.equal(outputs, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
