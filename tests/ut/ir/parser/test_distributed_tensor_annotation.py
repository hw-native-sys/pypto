# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Parser tests for ``pld.DistributedTensor[[shape], dtype]`` annotations (N1.2)."""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.pypto_core.ir import DistributedTensorType, TensorType


def test_distributed_tensor_param_resolves_to_distributed_type():
    @pl.program
    class P:
        @pl.function
        def f(self, x: pld.DistributedTensor[[256], pl.FP32]) -> pld.DistributedTensor[[256], pl.FP32]:
            return x

    gvar = P.get_global_var("f")
    assert gvar is not None
    func = P.functions[gvar]
    param_type = func.params[0].type
    assert isinstance(param_type, DistributedTensorType)
    assert isinstance(param_type, TensorType)  # subclass relationship preserved
    assert len(func.return_types) == 1
    assert isinstance(func.return_types[0], DistributedTensorType)


def test_plain_tensor_param_is_not_distributed_type():
    """``pl.Tensor[...]`` must not promote to ``DistributedTensorType``."""

    @pl.program
    class P:
        @pl.function
        def f(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

    gvar = P.get_global_var("f")
    assert gvar is not None
    func = P.functions[gvar]
    param_type = func.params[0].type
    assert isinstance(param_type, TensorType)
    assert not isinstance(param_type, DistributedTensorType)


def test_distributed_tensor_with_layout():
    """DistributedTensor mirrors Tensor's third-slot layout dispatch."""

    @pl.program
    class P:
        @pl.function
        def f(self, x: pld.DistributedTensor[[64], pl.FP32, pl.NZ]) -> pl.Tensor[[64], pl.FP32]:
            return x

    gvar = P.get_global_var("f")
    assert gvar is not None
    func = P.functions[gvar]
    param_type = func.params[0].type
    assert isinstance(param_type, DistributedTensorType)
    # Layout flows through TensorView, mirroring pl.Tensor[[shape], dtype, layout].
    assert param_type.tensor_view is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
