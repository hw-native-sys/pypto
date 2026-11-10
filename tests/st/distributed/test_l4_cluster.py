# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Generated CLUSTER/HOST/CHIP execution with global and owned local windows.

L4_TEST_CLUSTER_CONFIG must contain a JSON ClusterConfig object with explicit
nodes, python_executable, comm_profile and launcher settings. Device visibility,
MPI placement and rank launcher environments belong to the invoking test job.
Without that launch contract these tests do not start any workers.
"""

import json
import os
from dataclasses import replace

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto.backend import BackendType
from pypto.ir import ClusterConfig, ClusterNodeSpec
from pypto.runtime import RunConfig

SIZE = 64


def _cluster_config() -> ClusterConfig:
    raw = os.environ.get("L4_TEST_CLUSTER_CONFIG")
    if not raw:
        pytest.skip("L4_TEST_CLUSTER_CONFIG is required for explicit MPI/device placement")
    values = json.loads(raw)
    nodes = [ClusterNodeSpec(**node) for node in values.pop("nodes")]
    config = ClusterConfig(nodes=nodes, **values)
    assert len(config.nodes) == 2, "This test requires two explicitly placed MPI node entries"
    return config


def _program(local_devices: int, *, mixed: bool, dynamic_window: bool = False):
    nd = local_devices
    count_rows = 2 if mixed else 1

    @pl.jit.incore
    def global_step(
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        counts: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
    ) -> tuple[pl.Tensor[[1, SIZE], pl.FP32], pl.Tensor[[1, 1], pl.INT32]]:
        # Keep window writes and reads in the same InCore task. Only ordinary
        # outputs cross its return ABI; the runtime owns the window lifetime.
        data = pl.store(pl.load(inp, [0, 0], [1, SIZE]), [0, 0], data)
        out = pl.store(pl.load(data, [0, 0], [1, SIZE]), [0, 0], out)
        pl.write(counts, [0, 0], pld.nranks(pld.get_comm_ctx(data)))
        return out, counts

    @pl.jit
    def global_chip(
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        counts: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
    ) -> tuple[
        pld.DistributedTensor[[1, SIZE], pl.FP32],
        pl.Tensor[[1, SIZE], pl.FP32],
        pl.Tensor[[1, 1], pl.INT32],
    ]:
        out, counts = global_step(inp, data, out, counts)
        return data, out, counts

    @pl.jit.host
    def global_host(
        inputs: pl.Tensor[[nd, 1, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[nd, 1, SIZE], pl.FP32]],
        counts: pl.Out[pl.Tensor[[nd, 1, 1], pl.INT32]],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
    ) -> tuple[pl.Tensor[[nd, 1, SIZE], pl.FP32], pl.Tensor[[nd, 1, 1], pl.INT32]]:
        # The inherited window is an InOut resource. Its existing DT return
        # alias carries the post-call SSA value across the dispatch backedge;
        # it contributes no tensor slot to the runtime wire ABI.
        current_data = data
        for device, (current_data,) in pl.range(pld.num_workers(), init_values=(current_data,)):
            next_data, _output, _counts = global_chip(
                inputs[device], current_data, outputs[device], counts[device], worker=device
            )
            current_data = pl.yield_(next_data)
        return outputs, counts

    @pl.jit.incore
    def mixed_step(
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        local_data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        counts: pl.Out[pl.Tensor[[2, 1], pl.INT32]],
    ) -> tuple[pl.Tensor[[1, SIZE], pl.FP32], pl.Tensor[[2, 1], pl.INT32]]:
        data = pl.store(pl.load(inp, [0, 0], [1, SIZE]), [0, 0], data)
        local_data = pl.store(pl.load(data, [0, 0], [1, SIZE]), [0, 0], local_data)
        out = pl.store(pl.load(local_data, [0, 0], [1, SIZE]), [0, 0], out)
        pl.write(counts, [0, 0], pld.nranks(pld.get_comm_ctx(data)))
        pl.write(counts, [1, 0], pld.nranks(pld.get_comm_ctx(local_data)))
        return out, counts

    @pl.jit
    def mixed_chip(
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        local_data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        counts: pl.Out[pl.Tensor[[2, 1], pl.INT32]],
    ) -> tuple[
        pld.DistributedTensor[[1, SIZE], pl.FP32],
        pld.DistributedTensor[[1, SIZE], pl.FP32],
        pl.Tensor[[1, SIZE], pl.FP32],
        pl.Tensor[[2, 1], pl.INT32],
    ]:
        out, counts = mixed_step(inp, data, local_data, out, counts)
        return data, local_data, out, counts

    @pl.jit.host
    def mixed_host(
        inputs: pl.Tensor[[nd, 1, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[nd, 1, SIZE], pl.FP32]],
        counts: pl.Out[pl.Tensor[[nd, 2, 1], pl.INT32]],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
    ) -> tuple[pl.Tensor[[nd, 1, SIZE], pl.FP32], pl.Tensor[[nd, 2, 1], pl.INT32]]:
        local_buffer = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
        current_data = data
        for device, (current_data,) in pl.range(pld.num_workers(), init_values=(current_data,)):
            local_data = pld.window(local_buffer, [1, SIZE], dtype=pl.FP32)
            next_data, _local_data, _output, _counts = mixed_chip(
                inputs[device], current_data, local_data, outputs[device], counts[device], worker=device
            )
            current_data = pl.yield_(next_data)
        return outputs, counts

    host = mixed_host if mixed else global_host

    if dynamic_window:

        @pl.jit.pod
        def dynamic_pod(
            inputs: pl.Tensor[[2, nd, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[2, nd, 1, SIZE], pl.FP32]],
            counts: pl.Out[pl.Tensor[[2, nd, count_rows, 1], pl.INT32]],
            window_bytes: pl.Scalar[pl.INDEX],
        ):
            global_buffer = pld.alloc_window_buffer(window_bytes)
            for node in pl.range(pld.num_workers()):
                data = pld.window(global_buffer, [1, SIZE], dtype=pl.FP32)
                host(inputs[node], outputs[node], counts[node], data, worker=node)
            return outputs, counts

        return dynamic_pod

    @pl.jit.pod
    def pod(
        inputs: pl.Tensor[[2, nd, 1, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[2, nd, 1, SIZE], pl.FP32]],
        counts: pl.Out[pl.Tensor[[2, nd, count_rows, 1], pl.INT32]],
    ) -> tuple[pl.Tensor[[2, nd, 1, SIZE], pl.FP32], pl.Tensor[[2, nd, count_rows, 1], pl.INT32]]:
        global_buffer = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
        for node in pl.range(pld.num_workers()):
            data = pld.window(global_buffer, [1, SIZE], dtype=pl.FP32)
            host(inputs[node], outputs[node], counts[node], data, worker=node)
        return outputs, counts

    return pod


@pytest.mark.platforms("a2a3", "a2a3sim")
@pytest.mark.parametrize("mixed", [False, True], ids=["global", "mixed"])
def test_l4_global_window_stage_through(test_config, mixed, tmp_path):
    cluster = _cluster_config()
    if test_config.codegen_only:
        pytest.skip("This test requires real CHIP execution, not --codegen-only")
    nd = cluster.node_device_counts()[0]
    count_rows = 2 if mixed else 1
    inputs = torch.arange(2 * nd * SIZE, dtype=torch.float32).reshape(2, nd, 1, SIZE)
    outputs = torch.full_like(inputs, -1)
    counts = torch.full((2, nd, count_rows, 1), -1, dtype=torch.int32)
    config = RunConfig(
        backend_type=BackendType.Ascend910,
        execution_mode=test_config.execution_mode,
        cluster_config=cluster,
        save_kernels=True,
        save_kernels_dir=str(tmp_path / ("mixed" if mixed else "global")),
    )
    # Public compilation runs PTOAS and assembles the real recursive CHIP
    # closure before deployment. No transport, domain or kernel is substituted.
    compiled = _program(nd, mixed=mixed).compile(inputs, outputs, counts, config=config)
    expected_counts = torch.full_like(counts, sum(cluster.node_device_counts()))
    if mixed:
        expected_counts[:, :, 1, 0] = nd
    with compiled.prepare() as worker:
        for round_index in range(2):
            current = inputs + round_index * 1000
            outputs.fill_(-1)
            counts.fill_(-1)
            worker(current, outputs, counts)
            assert torch.equal(outputs, current)
            assert torch.equal(counts, expected_counts)


@pytest.mark.platforms("a2a3", "a2a3sim")
def test_l4_nonpersistent_multi_program_dynamic_windows(test_config, tmp_path):
    cluster = _cluster_config()
    if test_config.codegen_only:
        pytest.skip("This test requires real CHIP execution, not --codegen-only")
    nd = cluster.node_device_counts()[0]
    inputs = torch.arange(2 * nd * SIZE, dtype=torch.float32).reshape(2, nd, 1, SIZE)
    outputs = torch.empty_like(inputs)
    counts = torch.empty((2, nd, 1, 1), dtype=torch.int32)
    config = RunConfig(
        backend_type=BackendType.Ascend910,
        execution_mode=test_config.execution_mode,
        cluster_config=cluster,
        save_kernels=True,
    )
    programs = [
        _program(nd, mixed=False, dynamic_window=True).compile(
            inputs, outputs, counts, 256, config=replace(config, save_kernels_dir=str(tmp_path / f"p{i}"))
        )
        for i in range(2)
    ]
    with programs[0].prepare(extra_compiled=[programs[1]], persistent=False) as worker:
        for i, (index, size) in enumerate(((0, 256), (1, 512), (0, 768))):
            current = inputs + i * 1000
            outputs.fill_(-1)
            counts.fill_(-1)
            worker.run(programs[index], current, outputs, counts, size)
            assert torch.equal(outputs, current)
            assert torch.all(counts == sum(cluster.node_device_counts()))
