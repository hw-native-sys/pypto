# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank allreduce via the ``pld.tensor.allreduce`` intrinsic.

Same reduction result as ``test_l3_allreduce.py``, but the InCore body calls the
composite intrinsic ``pld.tensor.allreduce``. After ``LowerCompositeOps``
expands it into the chunked ready/read-complete-barrier protocol, this test
exercises the generated mesh implementation directly on board.

ST coverage: **P=2** and **P=4**, each at a non-aligned short length and a
length larger than UB. The test's own stage-in/out paths are chunked too, so
they do not hide allreduce failures behind an oversized helper tile.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir import DistributedConfig

STAGE_CHUNK = 8192


def _expected_allreduce(inputs: torch.Tensor, op_name: str = "sum") -> torch.Tensor:
    """Replicate the selected element-wise reduction on every rank."""
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
    size: int,
    *,
    op_name: str = "sum",
    torch_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Distinct per-rank tensors so the golden sum is non-trivial."""
    if op_name == "prod" or torch_dtype == torch.float16:
        rows = [
            (1.0 + r * 0.125 + torch.arange(size, dtype=torch.float32).remainder(5) * 0.0625).reshape(1, size)
            for r in range(n_ranks)
        ]
    else:
        rows = [
            torch.arange(r * 100.0, r * 100.0 + size, dtype=torch.float32).reshape(1, size)
            for r in range(n_ranks)
        ]
    return torch.stack(rows).to(torch_dtype)


def _build_allreduce_program(
    n_ranks: int,
    size: int,
    *,
    reduce_op: pld.ReduceOp = pld.ReduceOp.Sum,
    dtype=pl.FP32,
    dtype_bytes: int = 4,
):
    """Build an N-rank allreduce program at call time using the new intrinsic.

    Deferred construction lets this file collect even if the embedded body
    is rejected by the parser.
    """
    nr = n_ranks
    sz = size
    REDUCE_OP = reduce_op
    DTYPE = dtype
    DTYPE_BYTES = dtype_bytes
    # PTOAS treats [M, 1] tensors as DN/col-major and requires the physical
    # column byte size (rows * sizeof(dtype)) to be 32-byte aligned.  Pad the
    # degenerate helper tile to [8, 1] while retaining validshape=[1, 1].
    # Other lengths keep an aligned row-major stage chunk.
    stage_rows = 32 // dtype_bytes if size == 1 else 1
    if size == 1:
        stage_cols = 1
    elif dtype_bytes == 2:
        alignment = 32 // dtype_bytes
        stage_cols = min(STAGE_CHUNK, ((size + alignment - 1) // alignment) * alignment)
    else:
        stage_cols = STAGE_CHUNK

    @pl.program
    class AllReduceIntrinsicNRank:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            inp: pl.Tensor[[1, sz], DTYPE],
            out: pl.Out[pl.Tensor[[1, sz], DTYPE]],
            data: pl.InOut[pld.DistributedTensor[[1, sz], DTYPE]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[1, sz], DTYPE]:
            """One-call mesh allreduce via the ``pld.tensor.allreduce`` composite.

            ``LowerCompositeOps`` synthesises the ready barrier and the
            UB-sized remote_load+accumulate / barrier / store-back chunks.
            """
            # Phase 1 (stage-in) — copy local input into this rank's window slot.
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

            # Phases 2-4 (barrier + cross-rank reduce + write-back) — one call.
            # The composite rebinds `data` (in-place semantics, same as
            # ``pl.store``) so subsequent reads see the reduced slice.
            data = pld.tensor.allreduce(staged_data, signal, op=REDUCE_OP)

            # Stage-out — reduced chunks → local output.
            for col, (out_iter,) in pl.range(0, sz, stage_cols, init_values=(out,)):
                valid = pl.min(stage_cols, sz - col)
                acc = pl.load(
                    data,
                    [0, col],
                    [stage_rows, stage_cols],
                    valid_shape=[1, valid],
                )
                out_iter = pl.store(acc, [0, col], out_iter)
                staged_out = pl.yield_(out_iter)
            return staged_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, sz], DTYPE],
            out: pl.Out[pl.Tensor[[1, sz], DTYPE]],
            data: pl.InOut[pld.DistributedTensor[[1, sz], DTYPE]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[1, sz], DTYPE]:
            """Per-device orchestration wrapper around ``reduce_step``."""
            return self.reduce_step(inp, out, data, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, 1, sz], DTYPE],
            outputs: pl.Out[pl.Tensor[[nr, 1, sz], DTYPE]],
        ) -> pl.Tensor[[nr, 1, sz], DTYPE]:
            """Launch one chip orchestration per rank with shared window buffers.

            Signal buffer is allocated here (one cell per rank, INT32, zero-
            initialised by ``alloc_window_buffer``) and threaded into
            ``reduce_step`` as the cross-rank barrier — the intrinsic itself
            does not synthesise it; this matches the established pattern in
            ``test_l3_allreduce.py``.
            """
            data_buf = pld.alloc_window_buffer(sz * DTYPE_BYTES)
            signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [1, sz], dtype=DTYPE)
                signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
                self.chip_orch(
                    inputs[r],
                    outputs[r],
                    data,
                    signal,
                    device=r,
                )
            return outputs

    return AllReduceIntrinsicNRank


class TestL3TensorAllReduceIntrinsic:
    """L3 runtime coverage for arbitrary-length mesh allreduce lowering."""

    @pytest.mark.parametrize("size", [1, 17, 4097, 65537])
    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_allreduce_intrinsic(self, test_config, device_ids, n_ranks, size):
        """Run non-aligned and larger-than-UB mesh allreduce at P=2/P=4."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"allreduce P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_allreduce_program(n_ranks, size)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = _make_rank_inputs(n_ranks, size)
        outputs = torch.zeros((n_ranks, 1, size), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_allreduce(inputs)
        assert torch.allclose(outputs, expected), (
            f"allreduce intrinsic P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize(
        ("reduce_op", "op_name"),
        [
            (pld.ReduceOp.Sum, "sum"),
            (pld.ReduceOp.Max, "max"),
            (pld.ReduceOp.Min, "min"),
            (pld.ReduceOp.Prod, "prod"),
        ],
    )
    @pytest.mark.parametrize(
        ("dtype", "dtype_bytes", "torch_dtype"),
        [
            (pl.FP32, 4, torch.float32),
            (pl.FP16, 2, torch.float16),
        ],
    )
    def test_allreduce_reduce_ops(
        self,
        test_config,
        device_ids,
        reduce_op,
        op_name,
        dtype,
        dtype_bytes,
        torch_dtype,
    ):
        """Exercise every public reduction operator through the real pipeline."""
        n_ranks = 2
        size = 17
        if len(device_ids) < n_ranks:
            pytest.skip(f"allreduce P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_allreduce_program(
            n_ranks,
            size,
            reduce_op=reduce_op,
            dtype=dtype,
            dtype_bytes=dtype_bytes,
        )
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )
        inputs = _make_rank_inputs(
            n_ranks,
            size,
            op_name=op_name,
            torch_dtype=torch_dtype,
        )
        outputs = torch.zeros_like(inputs)
        compiled(inputs, outputs)
        expected = _expected_allreduce(inputs, op_name)
        tolerance = 2e-2 if torch_dtype == torch.float16 else 1e-5
        assert torch.allclose(outputs, expected, rtol=tolerance, atol=tolerance), (
            f"mesh {torch_dtype} {op_name} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("size", [1, 17, 8193])
    @pytest.mark.parametrize("n_ranks", [1, 2, 4])
    def test_allreduce_fp16_arbitrary_lengths(self, test_config, device_ids, size, n_ranks):
        """Cover short, odd-tail, and larger-than-UB FP16 inputs."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"allreduce P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_allreduce_program(
            n_ranks,
            size,
            dtype=pl.FP16,
            dtype_bytes=2,
        )
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )
        inputs = _make_rank_inputs(n_ranks, size, torch_dtype=torch.float16)
        outputs = torch.zeros_like(inputs)
        compiled(inputs, outputs)
        expected = _expected_allreduce(inputs)
        assert torch.allclose(outputs, expected, rtol=2e-2, atol=2e-2), (
            f"mesh FP16 length={size} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
