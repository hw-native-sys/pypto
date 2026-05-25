# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ``DistributedCompiledProgram.__call__`` argument acceptance.

These tests compile a small L3 program (no device needed, ``skip_ptoas=True``)
and mock ``execute_distributed`` so the calling convention can be exercised
without a Worker. The focus is the G1 widening: tensor parameters now accept a
worker-resident :class:`DeviceTensor` in addition to a host ``torch.Tensor``.
"""

from unittest.mock import patch

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
from pypto.runtime import DeviceTensor


@pl.program
class _L3AddProgram:
    """L3: HOST orch → CHIP worker (a + b → f)."""

    @pl.function(type=pl.FunctionType.InCore)
    def tile_add(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_f = pl.add(tile_a, tile_b)
        return pl.store(tile_f, [0, 0], f)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        return self.tile_add(a, b, f)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        return self.chip_orch(a, b, f)


@pytest.fixture
def compiled(tmp_path) -> DistributedCompiledProgram:
    prog = ir.compile(
        _L3AddProgram,
        output_dir=str(tmp_path),
        platform="a2a3sim",
        skip_ptoas=True,
        dump_passes=False,
    )
    assert isinstance(prog, DistributedCompiledProgram)
    return prog


def test_call_accepts_device_tensor(compiled):
    """A DeviceTensor input is accepted and passed through to execute_distributed."""
    a = torch.zeros(128, 128, dtype=torch.float32)
    weight = DeviceTensor(0xABCD0000, (128, 128), torch.float32)  # worker-resident
    out = torch.zeros(128, 128, dtype=torch.float32)

    with patch("pypto.runtime.distributed_runner.execute_distributed") as mock_exec:
        compiled(a, weight, out)

    mock_exec.assert_called_once()
    coerced = mock_exec.call_args.args[1]
    assert coerced[1] is weight  # DeviceTensor reached the runner unchanged


def test_call_rejects_non_tensor(compiled):
    """Non-tensor / non-DeviceTensor args still raise TypeError with guidance."""
    a = torch.zeros(128, 128, dtype=torch.float32)
    out = torch.zeros(128, 128, dtype=torch.float32)

    with patch("pypto.runtime.distributed_runner.execute_distributed"):
        with pytest.raises(TypeError, match="DeviceTensor"):
            compiled(a, "not a tensor", out)  # type: ignore[arg-type]


def test_call_validates_device_tensor_shape(compiled):
    """A DeviceTensor with the wrong shape is rejected by _validate_device_tensor."""
    a = torch.zeros(128, 128, dtype=torch.float32)
    bad = DeviceTensor(0xABCD0000, (64, 64), torch.float32)  # wrong shape
    out = torch.zeros(128, 128, dtype=torch.float32)

    with patch("pypto.runtime.distributed_runner.execute_distributed"):
        with pytest.raises(TypeError, match="shape"):
            compiled(a, bad, out)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
