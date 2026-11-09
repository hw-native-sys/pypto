# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reindex shape decisions may freeze; runtime/data-dependent decisions may not."""

import pypto.language as pl
import pytest
from pypto import passes
from pypto.pypto_core import passes as core_passes


@pytest.mark.parametrize("case", ["metadata", "num_tokens", "data", "incoming_data"])
def test_scalar_if_phi_requires_fixed_condition_and_both_fixed_inputs(case):
    condition = {
        "metadata": "width > capacity",
        "num_tokens": "num_tokens > capacity",
        "data": "pl.tensor.read(index, [0]) > capacity",
        "incoming_data": "width > capacity",
    }[case]
    yes = "pl.tensor.read(index, [0])" if case == "incoming_data" else "pl.cast(1, pl.INT32)"
    source = f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.Orchestration)
    def consume(
        self, c: pl.InOut[pl.Tensor[[80, 16], pl.FP32]], flag: pl.Scalar[pl.INT32]
    ) -> pl.Tensor[[80, 16], pl.FP32]:
        return c

    @pl.function(type=pl.FunctionType.Graph)
    def graph(
        self, c: pl.InOut[pl.Tensor[[80, 16], pl.FP32]], index: pl.Tensor[[1], pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[80, 16], pl.FP32]:
        width = pl.tensor.dim(c, 0)
        capacity = pl.tensor.dim(c, 1)
        flag = pl.cast(0, pl.INT32)
        if {condition}:
            flag = {yes}
        self.consume(c, flag)
        return c

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, c: pl.InOut[pl.Tensor[[80, 16], pl.FP32]], index: pl.Tensor[[1], pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[80, 16], pl.FP32]:
        self.graph(c, index, num_tokens)
        return c
"""
    program = pl.parse_program(source)
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        if case == "metadata":
            legalized = passes.legalize_graph_boundary()(passes.convert_to_ssa()(program))
            properties = core_passes.IRPropertySet()
            properties.insert(core_passes.IRProperty.GraphBoundaryLegalized)
            assert core_passes.PropertyVerifierRegistry.verify(properties, legalized) == []
        else:
            with pytest.raises(ValueError, match="value can differ between calls"):
                passes.legalize_graph_boundary()(passes.convert_to_ssa()(program))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
