# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 ST: mesh allgather with ``defer=True`` (split-phase).

``pld.tensor.allgather(..., defer=True)`` issues push+notify in a ``pl.at``
task and registers the barrier wait as ``pld.system.defer_wait``. A consumer
scope with ``deps=[ag_tid]`` reads the gathered window after completion.

Also covers signal hygiene: back-to-back deferred gathers on one signal
(unrolled, across an ``if``, and across persistent re-entry) must reflect
only the last inputs.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir import DistributedConfig

SIZE = 64
SECOND_CALL_OFFSET = 100000.0


def _expected_allgather(inputs: torch.Tensor) -> torch.Tensor:
    gathered = torch.cat([inputs[r, 0] for r in range(inputs.shape[0])])
    return torch.stack([gathered] * inputs.shape[0]).unsqueeze(1)


def _expected_allgather_from_locals(n_ranks: int, offset: float) -> torch.Tensor:
    rows = [
        torch.arange(offset + r * 100.0, offset + r * 100.0 + SIZE, dtype=torch.float32)
        for r in range(n_ranks)
    ]
    gathered = torch.stack(rows).reshape(n_ranks * SIZE)
    return gathered.unsqueeze(0).expand(n_ranks, 1, n_ranks * SIZE).contiguous()


def _make_rank_inputs(n_ranks: int, offset: float = 0.0) -> torch.Tensor:
    rows = [
        torch.arange(offset + r * 100.0, offset + r * 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE)
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


def _compile(program, test_config, device_ids, n_ranks):
    return ir.compile(
        program,
        platform=test_config.platform,
        distributed_config=DistributedConfig(
            device_ids=device_ids[:n_ranks],
            num_sub_workers=0,
        ),
    )


def _build_deferred_allgather_program(n_ranks: int):
    nr = n_ranks

    @pl.program
    class AllGatherDeferredNRank:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, nr * SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[1, nr * SIZE], pl.FP32]:
            with pl.manual_scope():
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="ag_defer") as ag_tid:
                    # Window-as-result: the call writes ``data`` in place. Do not
                    # rebind across the task edge — chip orch cannot forward a
                    # device-returned DistributedTensor alias between submits.
                    pld.tensor.allgather(inp, data, signal, defer=True)

                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="ag_consume",
                    deps=[ag_tid],
                ):
                    for r in pl.range(nr):
                        chunk = pl.load(data, [r, 0], [1, SIZE])
                        pl.store(chunk, [0, r * SIZE], out)
            return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, 1, nr * SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, 1, nr * SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(nr * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [nr, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], data, sig, device=r)
            return outputs

    return AllGatherDeferredNRank


def _build_deferred_allgather_twice(n_ranks: int):
    """Two deferred gathers on one signal; second inputs must win."""
    nr = n_ranks

    @pl.program
    class AllGatherDeferredTwice:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            first: pl.Tensor[[1, SIZE], pl.FP32],
            second: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, nr * SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[1, nr * SIZE], pl.FP32]:
            with pl.manual_scope():
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="ag_defer_0") as ag0:
                    pld.tensor.allgather(first, data, signal, defer=True)
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="ag_defer_1",
                    deps=[ag0],
                ) as ag1:
                    pld.tensor.allgather(second, data, signal, defer=True)
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="ag_consume",
                    deps=[ag1],
                ):
                    for r in pl.range(nr):
                        chunk = pl.load(data, [r, 0], [1, SIZE])
                        pl.store(chunk, [0, r * SIZE], out)
            return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            firsts: pl.Tensor[[nr, 1, SIZE], pl.FP32],
            seconds: pl.Tensor[[nr, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, 1, nr * SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, 1, nr * SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(nr * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [nr, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                self.chip_orch(firsts[r], seconds[r], outputs[r], data, sig, device=r)
            return outputs

    return AllGatherDeferredTwice


def _build_deferred_allgather_twice_across_if(n_ranks: int):
    """Second deferred gather sits behind a taken ``if``; signal still clears."""
    nr = n_ranks

    @pl.program
    class AllGatherDeferredTwiceAcrossIf:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            first: pl.Tensor[[1, SIZE], pl.FP32],
            second: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, nr * SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            take_second: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, nr * SIZE], pl.FP32]:
            with pl.manual_scope():
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="ag_defer_0") as ag0:
                    pld.tensor.allgather(first, data, signal, defer=True)
                # Always-true in the ST; keeps the second issue on a CFG edge.
                if take_second == 1:
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="ag_defer_1",
                        deps=[ag0],
                    ) as ag1:
                        pld.tensor.allgather(second, data, signal, defer=True)
                    consume_dep = ag1
                else:
                    consume_dep = ag0
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="ag_consume",
                    deps=[consume_dep],
                ):
                    for r in pl.range(nr):
                        chunk = pl.load(data, [r, 0], [1, SIZE])
                        pl.store(chunk, [0, r * SIZE], out)
            return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            firsts: pl.Tensor[[nr, 1, SIZE], pl.FP32],
            seconds: pl.Tensor[[nr, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, 1, nr * SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, 1, nr * SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(nr * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [nr, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                self.chip_orch(firsts[r], seconds[r], outputs[r], data, sig, 1, device=r)
            return outputs

    return AllGatherDeferredTwiceAcrossIf


class TestL3CompositeSplitPhaseAllGather:
    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_allgather_defer(self, test_config, device_ids, n_ranks):
        if len(device_ids) < n_ranks:
            pytest.skip(f"allgather defer P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = _compile(_build_deferred_allgather_program(n_ranks), test_config, device_ids, n_ranks)
        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, 1, n_ranks * SIZE), dtype=torch.float32)
        compiled(inputs, outputs)

        expected = _expected_allgather(inputs)
        assert torch.allclose(outputs, expected), (
            f"allgather defer=True P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("n_ranks", [2])
    def test_allgather_defer_back_to_back(self, test_config, device_ids, n_ranks):
        if len(device_ids) < n_ranks:
            pytest.skip(f"allgather defer P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = _compile(_build_deferred_allgather_twice(n_ranks), test_config, device_ids, n_ranks)
        firsts = _make_rank_inputs(n_ranks, 0.0)
        seconds = _make_rank_inputs(n_ranks, SECOND_CALL_OFFSET)
        outputs = torch.zeros((n_ranks, 1, n_ranks * SIZE), dtype=torch.float32)
        compiled(firsts, seconds, outputs)

        expected = _expected_allgather_from_locals(n_ranks, SECOND_CALL_OFFSET)
        assert torch.allclose(outputs, expected), (
            f"back-to-back allgather defer=True P={n_ranks} leaked first-call data: "
            f"max diff = {(outputs - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("n_ranks", [2])
    def test_allgather_defer_back_to_back_across_if(self, test_config, device_ids, n_ranks):
        if len(device_ids) < n_ranks:
            pytest.skip(f"allgather defer P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = _compile(
            _build_deferred_allgather_twice_across_if(n_ranks), test_config, device_ids, n_ranks
        )
        firsts = _make_rank_inputs(n_ranks, 0.0)
        seconds = _make_rank_inputs(n_ranks, SECOND_CALL_OFFSET)
        outputs = torch.zeros((n_ranks, 1, n_ranks * SIZE), dtype=torch.float32)
        compiled(firsts, seconds, outputs)

        expected = _expected_allgather_from_locals(n_ranks, SECOND_CALL_OFFSET)
        assert torch.allclose(outputs, expected), (
            f"if-path allgather defer=True P={n_ranks} leaked first-call data: "
            f"max diff = {(outputs - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("n_ranks", [2])
    def test_allgather_defer_persistent_reuse(self, test_config, device_ids, n_ranks, tmp_path):
        """Second invoke on the same windows must not see a sticky signal.

        Host-driven loop over a prepared worker with ``reset_persistent_windows=
        False`` — the in-kernel unrolled back-to-back ST covers the same
        hygiene inside one chip orch; this covers reuse across ``for``-like
        re-entries (signal-reuse gate).
        """
        if len(device_ids) < n_ranks:
            pytest.skip(f"allgather defer P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_deferred_allgather_program(n_ranks)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
            output_dir=str(tmp_path / "defer_persistent"),
        )

        firsts = _make_rank_inputs(n_ranks, 0.0).share_memory_()
        seconds = _make_rank_inputs(n_ranks, SECOND_CALL_OFFSET).share_memory_()
        outputs = torch.zeros((n_ranks, 1, n_ranks * SIZE), dtype=torch.float32).share_memory_()

        with compiled.prepare(
            config=test_config,
            persistent=True,
            reset_persistent_windows=False,
        ) as worker:
            worker(firsts, outputs, config=test_config)
            outputs.fill_(0)
            worker(seconds, outputs, config=test_config)

        expected = _expected_allgather_from_locals(n_ranks, SECOND_CALL_OFFSET)
        assert torch.allclose(outputs, expected), (
            f"persistent allgather defer=True P={n_ranks} leaked first-call data: "
            f"max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
