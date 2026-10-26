# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Runtime acceptance for descriptor-only pitched GM row sampling."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec


@pl.program
class SubviewRowsProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def copy_rows(
        self, cache: pl.Tensor[[8, 512], pl.BF16], output: pl.Out[pl.Tensor[[2, 512], pl.BF16]]
    ) -> pl.Tensor[[2, 512], pl.BF16]:
        selected = pl.tensor.subview(cache, [2, 512], [1, 0], steps=[3, 1])
        pair = pl.load(selected, [0, 0], [2, 512])
        pl.store(pair, [0, 0], output)
        return output

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, cache: pl.Tensor[[8, 512], pl.BF16], output: pl.Out[pl.Tensor[[2, 512], pl.BF16]]
    ) -> pl.Tensor[[2, 512], pl.BF16]:
        output = self.copy_rows(cache, output)
        return output


class SubviewRowsCase(PTOTestCase):
    __test__ = False

    def get_name(self) -> str:
        return "tensor_subview_rows_1_4"

    def define_tensors(self) -> list[TensorSpec]:
        # Values are BF16-exact and distinguish every row from every other row.
        values = 16 * torch.arange(8)[:, None] + torch.arange(512)[None, :] % 16
        return [
            TensorSpec("cache", [8, 512], DataType.BF16, init_value=values.to(torch.bfloat16)),
            TensorSpec("output", [2, 512], DataType.BF16, is_output=True),
        ]

    def get_program(self) -> Any:
        return SubviewRowsProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["output"][:] = tensors["cache"][[1, 4]]


@pytest.mark.parametrize("platform", ["a2a3sim", "a2a3"])
def test_tensor_subview_rows(test_runner, platform):
    result = test_runner.run(SubviewRowsCase(platform=platform))
    assert result.passed, result.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
