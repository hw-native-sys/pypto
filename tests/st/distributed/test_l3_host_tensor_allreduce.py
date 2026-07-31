# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST: host-orchestrator ``pld.tensor.allreduce`` builtin dispatch."""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 256
NR = pl.dynamic("NR")


def _expected_allreduce(inputs: torch.Tensor, op_name: str = "sum") -> torch.Tensor:
    if op_name == "sum":
        reduced = inputs.sum(dim=0)
    elif op_name == "max":
        reduced = inputs.max(dim=0).values
    elif op_name == "min":
        reduced = inputs.min(dim=0).values
    elif op_name == "prod":
        reduced = inputs.prod(dim=0)
    else:
        raise ValueError(f"unsupported golden allreduce op: {op_name}")
    return torch.stack([reduced] * inputs.shape[0])


def _make_rank_inputs(
    n_ranks: int,
    size: int = SIZE,
    *,
    dtype: torch.dtype = torch.float32,
    op_name: str = "sum",
) -> torch.Tensor:
    if op_name == "prod":
        rows = [
            (1.0 + r * 0.125 + torch.arange(size, dtype=torch.float32).remainder(5) * 0.0625).reshape(1, size)
            for r in range(n_ranks)
        ]
    else:
        rows = [
            torch.arange(r * 100.0, r * 100.0 + size, dtype=torch.float32).reshape(1, size)
            for r in range(n_ranks)
        ]
    return torch.stack(rows).to(dtype)


@pl.program
class HostTensorAllReduce:
    @pl.function(type=pl.FunctionType.InCore)
    def publish_step(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
        local = pl.load(inp, [0, 0], [1, SIZE])
        return pl.store(local, [0, 0], data)

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
        reduced = pl.load(data, [0, 0], [1, SIZE])
        return pl.store(reduced, [0, 0], out)

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
        signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())

        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
            self.publish_orch(inputs[r], data, device=r)

        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [pld.world_size()], dtype=pl.INT32)
        data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)

        for r in pl.range(pld.world_size()):
            self.consume_orch(data, outputs[r], device=r)

        return outputs


@pl.program
class HostTensorAllReduceMax:
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
        signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())

        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
            self.publish_orch(inputs[r], data, device=r)

        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [pld.world_size()], dtype=pl.INT32)
        data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Max)

        for r in pl.range(pld.world_size()):
            self.consume_orch(data, outputs[r], device=r)
        return outputs


def _build_host_allreduce(
    size: int,
    *,
    dtype,
    dtype_bytes: int,
    reduce_op: pld.ReduceOp = pld.ReduceOp.Sum,
):
    """Build a host-builtin allreduce with chunked stage-in/out helpers."""

    sz = size
    DTYPE = dtype
    DTYPE_BYTES = dtype_bytes
    REDUCE_OP = reduce_op
    alignment = 32 // dtype_bytes
    stage_rows = alignment if size == 1 else 1
    if size == 1:
        stage_cols = 1
    else:
        stage_cols = min(4096, ((size + alignment - 1) // alignment) * alignment)

    @pl.program
    class HostTensorAllReduceArbitraryLength:
        @pl.function(type=pl.FunctionType.InCore)
        def publish_step(
            self,
            inp: pl.Tensor[[1, sz], DTYPE],
            data: pl.InOut[pld.DistributedTensor[[1, sz], DTYPE]],
        ) -> pld.DistributedTensor[[1, sz], DTYPE]:
            for col, (data_iter,) in pl.range(0, sz, stage_cols, init_values=(data,)):
                valid = pl.min(stage_cols, sz - col)
                local = pl.load(
                    inp,
                    [0, col],
                    [stage_rows, stage_cols],
                    valid_shape=[1, valid],
                )
                data_iter = pl.store(local, [0, col], data_iter)
                staged_data = pl.yield_(data_iter)
            return staged_data

        @pl.function(type=pl.FunctionType.Orchestration)
        def publish_orch(
            self,
            inp: pl.Tensor[[1, sz], DTYPE],
            data: pl.InOut[pld.DistributedTensor[[1, sz], DTYPE]],
        ) -> pld.DistributedTensor[[1, sz], DTYPE]:
            return self.publish_step(inp, data)

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            data: pld.DistributedTensor[[1, sz], DTYPE],
            out: pl.Out[pl.Tensor[[1, sz], DTYPE]],
        ) -> pl.Tensor[[1, sz], DTYPE]:
            for col, (out_iter,) in pl.range(0, sz, stage_cols, init_values=(out,)):
                valid = pl.min(stage_cols, sz - col)
                reduced = pl.load(
                    data,
                    [0, col],
                    [stage_rows, stage_cols],
                    valid_shape=[1, valid],
                )
                out_iter = pl.store(reduced, [0, col], out_iter)
                staged_out = pl.yield_(out_iter)
            return staged_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def consume_orch(
            self,
            data: pld.DistributedTensor[[1, sz], DTYPE],
            out: pl.Out[pl.Tensor[[1, sz], DTYPE]],
        ) -> pl.Tensor[[1, sz], DTYPE]:
            return self.consume_step(data, out)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[NR, 1, sz], DTYPE],
            outputs: pl.Out[pl.Tensor[[NR, 1, sz], DTYPE]],
        ) -> pl.Tensor[[NR, 1, sz], DTYPE]:
            data_buf = pld.alloc_window_buffer(sz * DTYPE_BYTES)
            signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [1, sz], dtype=DTYPE)
                self.publish_orch(inputs[r], data, device=r)

            data = pld.window(data_buf, [1, sz], dtype=DTYPE)
            signal = pld.window(signal_buf, [pld.world_size()], dtype=pl.INT32)
            data = pld.tensor.allreduce(data, signal, op=REDUCE_OP)

            for r in pl.range(pld.world_size()):
                self.consume_orch(data, outputs[r], device=r)
            return outputs

    return HostTensorAllReduceArbitraryLength


class TestL3HostTensorAllReduce:
    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_host_tensor_allreduce(self, test_config, device_ids, n_ranks):
        if len(device_ids) < n_ranks:
            pytest.skip(f"host allreduce P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = ir.compile(
            HostTensorAllReduce,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        variant_dir = compiled.output_dir / "next_levels" / "builtin.tensor.allreduce__sum__fp32"
        assert variant_dir.is_dir()
        assert (variant_dir / "kernel_config.py").is_file()

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, 1, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_allreduce(inputs)
        assert torch.allclose(outputs, expected), (
            f"host allreduce P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

    def test_host_tensor_allreduce_max(self, test_config, device_ids):
        """Cover a non-default op in the materialized host builtin."""
        n_ranks = 2
        if len(device_ids) < n_ranks:
            pytest.skip(f"host allreduce P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = ir.compile(
            HostTensorAllReduceMax,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )
        variant_dir = compiled.output_dir / "next_levels" / "builtin.tensor.allreduce__max__fp32"
        assert variant_dir.is_dir()

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros_like(inputs)
        compiled(inputs, outputs)

        reduced = inputs.max(dim=0).values
        expected = torch.stack([reduced] * n_ranks)
        assert torch.equal(outputs, expected)

    @pytest.mark.parametrize("size", [1, 17, 257, 8193])
    def test_host_tensor_allreduce_fp16_arbitrary_lengths(self, test_config, device_ids, size):
        """Cover aligned transfer tails and more than one builtin tile."""
        n_ranks = 2
        if len(device_ids) < n_ranks:
            pytest.skip(f"host allreduce P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = ir.compile(
            _build_host_allreduce(size, dtype=pl.FP16, dtype_bytes=2),
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )
        variant_dir = compiled.output_dir / "next_levels" / "builtin.tensor.allreduce__sum__fp16"
        assert variant_dir.is_dir()

        inputs = _make_rank_inputs(n_ranks, size, dtype=torch.float16)
        outputs = torch.zeros_like(inputs)
        compiled(inputs, outputs)

        expected = _expected_allreduce(inputs)
        assert torch.allclose(outputs, expected, rtol=2e-2, atol=2e-2), (
            f"host FP16 length={size} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize(
        ("reduce_op", "op_name"),
        [
            (pld.ReduceOp.Max, "max"),
            (pld.ReduceOp.Min, "min"),
            (pld.ReduceOp.Prod, "prod"),
        ],
    )
    def test_host_tensor_allreduce_fp16_reduce_ops(
        self,
        test_config,
        device_ids,
        reduce_op,
        op_name,
    ):
        """Exercise non-default FP16 reductions through the host builtin."""
        n_ranks = 2
        size = 17
        if len(device_ids) < n_ranks:
            pytest.skip(f"host allreduce P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = ir.compile(
            _build_host_allreduce(
                size,
                dtype=pl.FP16,
                dtype_bytes=2,
                reduce_op=reduce_op,
            ),
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )
        variant_dir = compiled.output_dir / f"next_levels/builtin.tensor.allreduce__{op_name}__fp16"
        assert variant_dir.is_dir()

        inputs = _make_rank_inputs(
            n_ranks,
            size,
            dtype=torch.float16,
            op_name=op_name,
        )
        outputs = torch.zeros_like(inputs)
        compiled(inputs, outputs)

        expected = _expected_allreduce(inputs, op_name)
        assert torch.allclose(outputs, expected, rtol=2e-2, atol=2e-2), (
            f"host FP16 {op_name} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("size", [17, 257])
    def test_host_tensor_allreduce_fp32_arbitrary_lengths(self, test_config, device_ids, size):
        """Cover an odd tail and the first two-chunk FP32 input."""
        n_ranks = 2
        if len(device_ids) < n_ranks:
            pytest.skip(f"host allreduce P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = ir.compile(
            _build_host_allreduce(size, dtype=pl.FP32, dtype_bytes=4),
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )
        inputs = _make_rank_inputs(n_ranks, size)
        outputs = torch.zeros_like(inputs)
        compiled(inputs, outputs)

        expected = _expected_allreduce(inputs)
        assert torch.equal(outputs, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
