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


def _program(local_devices: int, *, mixed: bool):
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


def _shared_programs(local_devices: int, rank_count: int):
    """Two actual JIT programs with different window names and one run per phase."""
    nd, nr = local_devices, rank_count

    @pl.jit.incore
    def add_to_window(
        lhs: pl.Tensor[[1, SIZE], pl.FP32],
        rhs: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        echo: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        value = pl.add(pl.load(lhs, [0, 0], [1, SIZE]), pl.load(rhs, [0, 0], [1, SIZE]))
        data = pl.store(value, [0, 0], data)
        return pl.store(value, [0, 0], echo)

    @pl.jit
    def compute_chip(
        lhs: pl.Tensor[[1, SIZE], pl.FP32],
        rhs: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        echo: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    ) -> tuple[pld.DistributedTensor[[1, SIZE], pl.FP32], pl.Tensor[[1, SIZE], pl.FP32]]:
        echo = add_to_window(lhs, rhs, data, echo)
        return data, echo

    @pl.jit.host
    def compute_host(
        lhs: pl.Tensor[[nd, 1, SIZE], pl.FP32],
        rhs: pl.Tensor[[nd, 1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        echo: pl.Out[pl.Tensor[[nd, 1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[nd, 1, SIZE], pl.FP32]:
        current_data = data
        for device, (current_data,) in pl.range(pld.num_workers(), init_values=(current_data,)):
            next_data, _echo = compute_chip(
                lhs[device], rhs[device], current_data, echo[device], worker=device
            )
            current_data = pl.yield_(next_data)
        return echo

    @pl.jit.pod
    def compute(
        lhs: pl.Tensor[[2, nd, 1, SIZE], pl.FP32],
        rhs: pl.Tensor[[2, nd, 1, SIZE], pl.FP32],
        echo: pl.Out[pl.Tensor[[2, nd, 1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[2, nd, 1, SIZE], pl.FP32]:
        prefill_window = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
        for node in pl.range(pld.num_workers()):
            data = pld.window(prefill_window, [1, SIZE], dtype=pl.FP32)
            compute_host(lhs[node], rhs[node], data, echo[node], worker=node)
        return echo

    @pl.jit.incore
    def read_every_peer(
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        out: pl.Out[pl.Tensor[[nr, SIZE], pl.FP32]],
    ) -> pl.Tensor[[nr, SIZE], pl.FP32]:
        ctx = pld.get_comm_ctx(data)
        current_output = out
        for peer, (current_output,) in pl.range(pld.nranks(ctx), init_values=(current_output,)):
            value = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, SIZE])
            current_output = pl.yield_(pl.store(value, [peer, 0], current_output))
        return current_output

    @pl.jit
    def communication_chip(
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        out: pl.Out[pl.Tensor[[nr, SIZE], pl.FP32]],
    ) -> tuple[pld.DistributedTensor[[1, SIZE], pl.FP32], pl.Tensor[[nr, SIZE], pl.FP32]]:
        out = read_every_peer(data, out)
        return data, out

    @pl.jit.host
    def communication_host(
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        outputs: pl.Out[pl.Tensor[[nd, nr, SIZE], pl.FP32]],
    ) -> pl.Tensor[[nd, nr, SIZE], pl.FP32]:
        current_data = data
        for device, (current_data,) in pl.range(pld.num_workers(), init_values=(current_data,)):
            next_data, _output = communication_chip(current_data, outputs[device], worker=device)
            current_data = pl.yield_(next_data)
        return outputs

    @pl.jit.pod
    def communication(
        outputs: pl.Out[pl.Tensor[[2, nd, nr, SIZE], pl.FP32]],
    ) -> pl.Tensor[[2, nd, nr, SIZE], pl.FP32]:
        decode_window = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
        for node in pl.range(pld.num_workers()):
            data = pld.window(decode_window, [1, SIZE], dtype=pl.FP32)
            communication_host(data, outputs[node], worker=node)
        return outputs

    return compute, communication


def _run_shared_programs(test_config, tmp_path, *, cold: bool = False, default_reset: bool = False):
    from pypto import ir  # noqa: PLC0415

    cluster = _cluster_config()
    if test_config.codegen_only:
        pytest.skip("This test requires real CHIP execution, not --codegen-only")
    nd = cluster.node_device_counts()[0]
    nranks = sum(cluster.node_device_counts())
    inputs = torch.arange(nranks * SIZE, dtype=torch.float32).reshape(2, nd, 1, SIZE)
    rhs = torch.full_like(inputs, 3)
    echo = torch.full_like(inputs, -1)
    peer_values = torch.full((2, nd, nranks, SIZE), -1, dtype=torch.float32)
    config = RunConfig(
        backend_type=BackendType.Ascend910,
        execution_mode=test_config.execution_mode,
        cluster_config=cluster,
        save_kernels=True,
        save_kernels_dir=str(tmp_path / "two-programs"),
    )
    compute_jit, communication_jit = _shared_programs(nd, nranks)
    compute = compute_jit.compile(
        inputs,
        rhs,
        echo,
        config=replace(config, save_kernels_dir=str(tmp_path / "two-programs" / "compute")),
    )
    communication = communication_jit.compile(
        peer_values,
        config=replace(config, save_kernels_dir=str(tmp_path / "two-programs" / "communication")),
    )
    assert compute.output_dir != communication.output_dir, "Each compiled program must own its artifacts"
    if cold:
        compute_domains, communication_domains = compute.comm_domains, communication.comm_domains
        compute_platform, communication_platform = compute.platform, communication.platform
        # Execute only metadata-restored objects: neither can consult post-pass IR
        # when validating sharing or assembling the generated runtime artifacts.
        compute = ir.DistributedCompiledProgram.from_dir(
            compute.output_dir,
            backend_type=BackendType.Ascend910,
            execution_mode=test_config.execution_mode,
        )
        communication = ir.DistributedCompiledProgram.from_dir(
            communication.output_dir,
            backend_type=BackendType.Ascend910,
            execution_mode=test_config.execution_mode,
        )
        assert compute._program is None
        assert communication._program is None
        assert compute.platform == compute_platform
        assert communication.platform == communication_platform
        assert compute.cluster_config == communication.cluster_config == cluster
        assert compute.comm_domains == compute_domains
        assert communication.comm_domains == communication_domains
    binding = getattr(ir, "SharedDomainBinding", None)
    assert callable(binding), "I2 requires public explicit session sharing"
    assert len(compute.comm_domains) == len(communication.comm_domains) == 1
    assert (
        compute.comm_domains[0]["slots"][0]["generated_name"]
        != (communication.comm_domains[0]["slots"][0]["generated_name"])
    )
    # Omit the reset option to exercise the public default, rather than passing
    # True and accidentally leaving the None -> True contract untested.
    reset_options = {} if default_reset else {"reset_persistent_windows": False}
    with compute.prepare(
        extra_compiled=[communication],
        persistent=True,
        shared_domains=[
            binding(program=compute, domain=0, key="activations", slots={0: "values"}),
            binding(program=communication, domain=0, key="activations", slots={0: "values"}),
        ],
        **reset_options,
    ) as worker:
        for iteration in range(2):
            current = inputs + iteration * 1000
            echo.fill_(-1)
            peer_values.fill_(-1)
            worker.run(compute, current, rhs, echo)
            assert torch.equal(echo, current + rhs)
            worker.run(communication, peer_values)
            expected = (current + rhs).reshape(nranks, SIZE)
            if default_reset:
                # Compute really wrote the changed sum (echo above); the next
                # run must clear every shared peer window before any TLOAD.
                expected = torch.zeros_like(expected)
            assert torch.equal(peer_values, expected.expand_as(peer_values))


@pytest.mark.platforms("a2a3", "a2a3sim")
def test_i2_l4_two_program_compute_then_tload_every_rank(test_config, tmp_path):
    _run_shared_programs(test_config, tmp_path)


@pytest.mark.platforms("a2a3", "a2a3sim")
def test_i3_l4_cold_two_program_compute_then_tload_every_rank(test_config, tmp_path):
    _run_shared_programs(test_config, tmp_path, cold=True)


@pytest.mark.platforms("a2a3", "a2a3sim")
def test_i3_l4_shared_domains_default_reset_every_rank(test_config, tmp_path):
    _run_shared_programs(test_config, tmp_path, default_reset=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
