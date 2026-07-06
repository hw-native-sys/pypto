# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
External C++ Kernel Integration System Test.

Exercises calling a hand-written AICore kernel from a PyPTO orchestration via
``@pl.function(type=AIV, external_source=...)``: the DSL declares only the
signature (empty ``...`` body), the compiler skips codegen for it and compiles
the referenced ``kernels/aiv/spmd_write.cpp`` as the InCore kernel, and the
PyPTO orchestration dispatches it across SPMD blocks.

The kernel writes ``float(block_idx)`` at ``out[(base_cl + block_idx) * 16]``,
so a 4-block SPMD launch with ``base_cl = 0`` produces ``[0, 1, 2, 3]`` at
stride 16 — an exact, deterministic golden.
"""

from pathlib import Path
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

_KERNEL = Path(__file__).parent / "kernels" / "aiv" / "spmd_write.cpp"

_CORE_NUM = 4
_BASE_CL = 0
_FLOATS_PER_CACHE_LINE = 16
# Room for cache lines [BASE_CL, BASE_CL + CORE_NUM); pad to a full extra line.
_TOTAL = (_BASE_CL + _CORE_NUM + 1) * _FLOATS_PER_CACHE_LINE


@pl.program
class SpmdWriteExternProgram:
    """PyPTO orchestration driving a hand-written external AIV kernel."""

    @pl.function(type=pl.FunctionType.AIV, external_source=_KERNEL)
    def SPMD_WRITE_AIV(
        self,
        out: pl.InOut[pl.Tensor[[_TOTAL], pl.FP32]],
        base_cl: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[_TOTAL], pl.FP32]: ...  # implementation lives in kernels/aiv/spmd_write.cpp

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, out: pl.Out[pl.Tensor[[_TOTAL], pl.FP32]]) -> pl.Tensor[[_TOTAL], pl.FP32]:
        with pl.spmd(core_num=_CORE_NUM):
            out = self.SPMD_WRITE_AIV(out, _BASE_CL)
        return out


class SpmdWriteExternTest(PTOTestCase):
    """External AIV kernel dispatched across 4 SPMD blocks; exact golden."""

    __test__ = False

    def get_name(self) -> str:
        return "external_kernel_spmd_write_aiv"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("out", [_TOTAL], DataType.FP32, init_value=0.0, is_output=True),
        ]

    def get_program(self) -> Any:
        return SpmdWriteExternProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        out = torch.zeros(_TOTAL, dtype=torch.float32)
        for block_idx in range(_CORE_NUM):
            out[(_BASE_CL + block_idx) * _FLOATS_PER_CACHE_LINE] = float(block_idx)
        tensors["out"][:] = out


class TestExternalKernel:
    """End-to-end integration of a hand-written C++ kernel via external_source."""

    def test_external_aiv_kernel(self, test_runner):
        """A PyPTO orchestration dispatches an external AIV kernel across SPMD blocks."""
        result = test_runner.run(SpmdWriteExternTest())
        assert result.passed, f"External AIV kernel failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
