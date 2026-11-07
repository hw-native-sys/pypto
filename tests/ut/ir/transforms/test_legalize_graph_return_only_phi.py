# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Exact return-only tensor merges, including Out and Submit aliases."""

import pypto.language as pl
import pytest
from pypto import passes
from pypto.ir.printer import python_print

N = pl.dynamic("alias_profile_rows")


@pl.program
class SameCapture:
    @pl.function
    def touch(self, target: pl.InOut[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            tile = pl.load(target, [0, 0], [16, 16])
            pl.store(tile, [0, 0], target)
        return target

    @pl.function(type=pl.FunctionType.Graph)
    def layer(
        self, target: pl.InOut[pl.Tensor[[16, 16], pl.FP32]], profile: pl.Tensor[[N, 16], pl.FP32]
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        if pl.tensor.dim(profile, 0) > 8:
            merged = target
        else:
            merged = self.touch(target)
        return merged

    @pl.function
    def main(
        self, target: pl.InOut[pl.Tensor[[16, 16], pl.FP32]], profile: pl.Tensor[[N, 16], pl.FP32]
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        return self.layer(target, profile)


@pl.program
class DifferentCapture:
    @pl.function(type=pl.FunctionType.Graph)
    def layer(
        self,
        left: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        right: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        profile: pl.Tensor[[N, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        if pl.tensor.dim(profile, 0) > 8:
            merged = left
        else:
            merged = right
        return merged

    @pl.function
    def main(
        self,
        left: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        right: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        profile: pl.Tensor[[N, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        return self.layer(left, right, profile)


@pl.program
class ChangedView:
    @pl.function(type=pl.FunctionType.Graph)
    def layer(
        self, target: pl.InOut[pl.Tensor[[16, 16], pl.FP32]], profile: pl.Tensor[[N, 16], pl.FP32]
    ) -> pl.Tensor[[8, 16], pl.FP32]:
        if pl.tensor.dim(profile, 0) > 8:
            merged = target
        else:
            merged = target
        view = pl.tensor.slice(merged, [8, 16], [0, 0])
        return view

    @pl.function
    def main(
        self, target: pl.InOut[pl.Tensor[[16, 16], pl.FP32]], profile: pl.Tensor[[N, 16], pl.FP32]
    ) -> pl.Tensor[[8, 16], pl.FP32]:
        return self.layer(target, profile)


@pl.program
class MovingBoundary:
    @pl.function(type=pl.FunctionType.Graph)
    def layer(
        self,
        target: pl.InOut[pl.Tensor[[32, 16], pl.FP32]],
        dest: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        profile: pl.Tensor[[N, 16], pl.FP32],
        offset: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[32, 16], pl.FP32]:
        if pl.tensor.dim(profile, 0) > 8:
            merged = target
        else:
            merged = target
        view = pl.tensor.slice(merged, [16, 16], [offset, 0])
        with pl.at(level=pl.Level.CORE_GROUP):
            tile = pl.load(view, [0, 0], [16, 16])
            pl.store(tile, [0, 0], dest)
        return merged

    @pl.function
    def main(
        self,
        target: pl.InOut[pl.Tensor[[32, 16], pl.FP32]],
        dest: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        profile: pl.Tensor[[N, 16], pl.FP32],
        offset: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[32, 16], pl.FP32]:
        return self.layer(target, dest, profile, offset)


@pl.program
class NestedTupleCaptures:
    @pl.function
    def touch(
        self, first: pl.InOut[pl.Tensor[[16, 16], pl.FP32]], second: pl.InOut[pl.Tensor[[8, 16], pl.FP32]]
    ) -> pl.Tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[8, 16], pl.FP32]]:
        with pl.at(level=pl.Level.CORE_GROUP):
            a = pl.load(first, [0, 0], [16, 16])
            pl.store(a, [0, 0], first)
        with pl.at(level=pl.Level.CORE_GROUP):
            b = pl.load(second, [0, 0], [8, 16])
            pl.store(b, [0, 0], second)
        return first, second

    @pl.function(type=pl.FunctionType.Graph)
    def layer(
        self,
        first: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        second: pl.InOut[pl.Tensor[[8, 16], pl.FP32]],
        profile: pl.Tensor[[N, 16], pl.FP32],
    ) -> pl.Tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[8, 16], pl.FP32]]:
        if pl.tensor.dim(profile, 0) > 8:
            left = first
            right = second
        else:
            left, right = self.touch(first, second)
        return left, right

    @pl.function
    def main(
        self,
        first: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        second: pl.InOut[pl.Tensor[[8, 16], pl.FP32]],
        profile: pl.Tensor[[N, 16], pl.FP32],
    ) -> pl.Tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[8, 16], pl.FP32]]:
        return self.layer(first, second, profile)


# Match actual pool_select_spmd: two flat return positions, not one nested TupleType.
@pl.program
class SameTwoCaptures:
    @pl.function
    def touch(
        self, first: pl.InOut[pl.Tensor[[16, 16], pl.FP32]], second: pl.InOut[pl.Tensor[[8, 16], pl.FP32]]
    ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[8, 16], pl.FP32]]:
        with pl.at(level=pl.Level.CORE_GROUP):
            a = pl.load(first, [0, 0], [16, 16])
            pl.store(a, [0, 0], first)
        with pl.at(level=pl.Level.CORE_GROUP):
            b = pl.load(second, [0, 0], [8, 16])
            pl.store(b, [0, 0], second)
        return first, second

    @pl.function(type=pl.FunctionType.Graph)
    def layer(
        self,
        first: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        second: pl.InOut[pl.Tensor[[8, 16], pl.FP32]],
        profile: pl.Tensor[[N, 16], pl.FP32],
    ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[8, 16], pl.FP32]]:
        if pl.tensor.dim(profile, 0) > 8:
            left = first
            right = second
        else:
            left, right = self.touch(first, second)
        return left, right

    @pl.function
    def main(
        self,
        first: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        second: pl.InOut[pl.Tensor[[8, 16], pl.FP32]],
        profile: pl.Tensor[[N, 16], pl.FP32],
    ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[8, 16], pl.FP32]]:
        return self.layer(first, second, profile)


def _legalize(program):
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        ssa = passes.convert_to_ssa()(program)
        outlined = passes.outline_incore_scopes()(ssa)
        normalized = passes.normalize_return_order()(outlined)
        return passes.legalize_graph_boundary()(normalized)


@pytest.mark.parametrize("program,count", [(SameCapture, 1), (SameTwoCaptures, 2)])
def test_exact_return_only_inout_phi_is_own_parameter(program, count):
    result = _legalize(program)
    graph = result.get_function("layer")
    assert graph is not None
    printed = python_print(graph)
    expected = "return " + ", ".join(p.name_hint for p in graph.params[:count])
    assert expected in printed, printed
    assert "touch(" in printed, printed  # The branch call and writes must survive.
    assert "if " in printed, printed


@pytest.mark.parametrize("program", [DifferentCapture, ChangedView, MovingBoundary, NestedTupleCaptures])
def test_unknown_descriptor_or_nonreturn_phi_remains_rejected(program):
    with pytest.raises(ValueError, match="returns a value it computed"):
        _legalize(program)


@pl.program
class SameOutCapture:
    @pl.function
    def write(self, target: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            tile = pl.create_tile([16, 16], dtype=pl.FP32)
            pl.store(tile, [0, 0], target)
        return target

    @pl.function(type=pl.FunctionType.Graph)
    def layer(
        self, target: pl.Out[pl.Tensor[[16, 16], pl.FP32]], profile: pl.Tensor[[N, 16], pl.FP32]
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        if pl.tensor.dim(profile, 0) > 8:
            merged = self.write(target)
        else:
            merged = self.write(target)
        return merged

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, target: pl.Out[pl.Tensor[[16, 16], pl.FP32]], profile: pl.Tensor[[N, 16], pl.FP32]
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        self.layer(target, profile)
        return target


@pl.program
class SameSubmitCaptures:
    @pl.function(type=pl.FunctionType.Orchestration)
    def write(
        self, first: pl.InOut[pl.Tensor[[16, 16], pl.FP32]], second: pl.Out[pl.Tensor[[8, 16], pl.INT32]]
    ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[8, 16], pl.INT32]]:
        return first, second

    @pl.function(type=pl.FunctionType.Graph)
    def layer(
        self,
        first: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        second: pl.Out[pl.Tensor[[8, 16], pl.INT32]],
        profile: pl.Tensor[[N, 16], pl.FP32],
    ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[8, 16], pl.INT32]]:
        with pl.manual_scope():
            if pl.tensor.dim(profile, 0) > 8:
                (left, right), done = pl.submit(self.write, first, second)
            else:
                (left, right), done = pl.submit(self.write, first, second)
        return left, right

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        first: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        second: pl.Out[pl.Tensor[[8, 16], pl.INT32]],
        profile: pl.Tensor[[N, 16], pl.FP32],
    ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[8, 16], pl.INT32]]:
        self.layer(first, second, profile)
        return first, second


@pytest.mark.parametrize("program,count", [(SameOutCapture, 1), (SameSubmitCaptures, 2)])
def test_written_capture_phi_preserves_exact_return_alias(program, count):
    graph = _legalize(program).get_function("layer")
    assert graph is not None
    printed = python_print(graph)
    assert "return " + ", ".join(p.name_hint for p in graph.params[:count]) in printed
    assert "if " in printed and "write" in printed
    if program is SameSubmitCaptures:
        assert printed.count("pl.submit(write,") == 2, printed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
