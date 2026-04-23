# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ST test for ``pl.at(name_hint=...)`` on a CORE_GROUP scope.

Verifies that a ``pl.at`` region annotated with
``level=pl.Level.CORE_GROUP, name_hint="GetKVCache"`` compiles and runs
end-to-end. The supplied ``name_hint`` should propagate to the outlined
function name and downstream artifacts (e.g. merged swimlane JSON).
See issue #1113.
"""

import sys
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType

M = 32
K = 64
N = 32

_NAME_HINT = "GetKVCache"


@pl.program
class AtNameHintProgram:
    """CORE_GROUP scope annotated with ``name_hint="GetKVCache"``."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="GetKVCache"):
            out = pl.matmul(a, b)
            output = pl.assemble(output, out, [0, 0])
        return output


class AtNameHintTest(PTOTestCase):
    """Compile and verify CORE_GROUP scope with name_hint."""

    __test__ = False

    def get_name(self) -> str:
        return "at_name_hint_get_kv_cache"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [K, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return AtNameHintProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["output"][:] = torch.matmul(tensors["a"].float(), tensors["b"].float())


class TestAtNameHint:
    """Regression test for issue #1113 — ``name_hint`` on CORE_GROUP scopes."""

    def test_core_group_name_hint(self, test_runner):
        """CORE_GROUP region with ``name_hint`` compiles and runs successfully."""
        result = test_runner.run(AtNameHintTest(backend_type=BackendType.Ascend910B))
        assert result.passed, f"pl.at(name_hint='{_NAME_HINT}') run failed: {result.error}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
