# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Prove only fixed metadata-view dimensions; retain all data-dependent guards."""

import re

import pypto.language as pl
import pytest
from pypto import ir, passes


def _legalize(program: ir.Program) -> ir.Program:
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        return passes.legalize_graph_boundary()(passes.convert_to_ssa()(program))


def _legalize_outlined(program: ir.Program) -> ir.Program:
    """``_legalize`` with the scope outliner in front, as the real pipeline has it.

    ``OutlineIncoreScopes`` rewrites an in-place device scope into a call, which
    rebinds the InOut parameter the body writes through::

        c_1 = layer_incore_0(a, c)
        return c_1

    That is the shape every Graph body actually has by the time this pass runs,
    so a check written against the pre-outlining shape can reject the entire
    feature while ``_legalize`` still passes.

    ``NormalizeReturnOrder`` is in the chain for the same reason. It runs at
    pass 28 and this pass at 47, and it is what establishes
    ``IRProperty::ReturnParamsExplicit`` — the pointer-identity return -> param
    map the pass reads to carry boundary provenance across an in-place call
    (``tmp = kernel(a, tmp)``). Without it that map answers nullopt for every
    callee, the propagation cannot fire, and a test asserting on it passes for
    the wrong reason.
    """
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        ssa = passes.convert_to_ssa()(program)
        outlined = passes.outline_incore_scopes()(ssa)
        return passes.legalize_graph_boundary()(passes.normalize_return_order()(outlined))


def _graph_func(program: ir.Program, name: str) -> ir.Function:
    func = program.get_function(name)
    assert func is not None
    return func


def _scalar_param_names(func: ir.Function) -> list[str]:
    """Scalar parameter names with ConvertToSSA's ``__ssa_vN`` suffix stripped."""
    return [re.sub(r"__ssa_v\d+$", "", p.name_hint) for p in func.params if isinstance(p.type, ir.ScalarType)]


def test_an_invariant_view_dim_read_is_accepted():
    """A constant slab keeps its extent when its invariant offset changes.

    The view must remain inside the loop because its induction variable has
    no call-site name. Its shape, and a reshape of it, still reproduce the
    same dimensions on every replay. This is the packed-workspace pattern.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.Graph)
        def view_dim(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            for i in pl.range(4):
                slab = pl.slice(a, [128, 128], [i * 128, 0])
                matrix = pl.reshape(slab, [128, 128])
                matrix_view = pl.tensor.view(matrix, [128, 128])
                view_rows = pl.tensor.dim(matrix_view, 0)
                with pl.at(level=pl.Level.CORE_GROUP):
                    tile: pl.Tile[[128, 128], pl.FP32] = pl.load(matrix, [view_rows - 128, 0], [128, 128])
                    pl.store(tile, [0, 0], c)
            return c

        @pl.function
        def main(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            self.view_dim(a, c)
            return c

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def view_dim_incore_0(
            self,
            matrix: pl.Tensor[[128, 128], pl.FP32],
            view_rows: pl.Scalar[pl.INDEX],
            target: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            tile: pl.Tile[[128, 128], pl.FP32] = pl.tile.load(
                matrix, [view_rows - 128, 0], [128, 128], [128, 128]
            )
            _stored = pl.tile.store(tile, [0, 0], target)
            return target

        @pl.function(type=pl.FunctionType.Graph, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def view_dim(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            for i, (carry,) in pl.range(4, init_values=(c,)):
                slab = pl.tensor.slice(a, [128, 128], [i * 128, 0])
                matrix = pl.tensor.reshape(slab, [128, 128])
                matrix_view = pl.tensor.view(matrix, [128, 128])
                view_rows = pl.tensor.dim(matrix_view, 0)
                updated = self.view_dim_incore_0(matrix, view_rows, carry)
                _result = pl.yield_(updated)
            return c

        @pl.function
        def main(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            self.view_dim(a, c)
            return c

    ir.assert_structural_equal(_legalize_outlined(Before), Expected)


def test_an_invariant_view_dim_can_feed_a_submit():
    """Submit arguments require the same shape proof as ordinary calls."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.Orchestration)
        def consume(
            self,
            a: pl.Tensor[[128, 128], pl.FP32],
            c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            rows: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            return c

        @pl.function(type=pl.FunctionType.Graph)
        def view_submit(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            with pl.manual_scope():
                for i in pl.range(4):
                    slab = pl.slice(a, [128, 128], [i * 128, 0])
                    rows = pl.tensor.dim(slab, 0)
                    c, done = pl.submit(self.consume, slab, c, rows)
            return c

    graph = _graph_func(_legalize(Before), "view_submit")
    assert "rows" not in _scalar_param_names(graph)


def test_a_data_dependent_view_offset_is_still_rejected():
    """Knowing a view's extent must not bless its varying storage offset."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.Graph)
        def varying_offset(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            index: pl.Tensor[[1], pl.INT32],
            c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            offset = pl.read(index, [0])
            slab = pl.slice(a, [128, 128], [offset, 0])
            rows = pl.tensor.dim(slab, 0)
            with pl.at(level=pl.Level.CORE_GROUP):
                tile: pl.Tile[[128, 128], pl.FP32] = pl.load(slab, [0, 0], [128, 128])
                pl.store(tile, [rows - 128, 0], c)
            return c

    with pytest.raises(ValueError, match="neither reconstructible"):
        _legalize_outlined(Before)


def test_a_data_dependent_view_shape_is_still_rejected():
    """A shape read from device data cannot become a frozen extent."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.Graph)
        def varying_shape(
            self,
            a: pl.Tensor[[512, 128], pl.FP32],
            index: pl.Tensor[[1], pl.INT32],
            c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            extent = pl.read(index, [0])
            for i in pl.range(4):
                slab = pl.slice(a, [extent, 128], [i * 128, 0])
                rows = pl.tensor.dim(slab, 0)
                with pl.at(level=pl.Level.CORE_GROUP):
                    tile: pl.Tile[[128, 128], pl.FP32] = pl.load(slab, [0, 0], [128, 128])
                    pl.store(tile, [rows - 128, 0], c)
            return c

    with pytest.raises(ValueError, match="neither reconstructible|shape is not"):
        _legalize_outlined(Before)


def test_c1_capacity_uses_constant_trip_view_extent():
    # Actual C1 o_proj shape lineage: the physical capacity is 80, independent
    # of runtime num_tokens. Keep the loop index and view descriptor in scope.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.Graph)
        def c1_group(
            self,
            a: pl.Tensor[[80, 128], pl.FP32],
            c: pl.InOut[pl.Tensor[[80, 128], pl.FP32]],
        ) -> pl.Tensor[[80, 128], pl.FP32]:
            for group_start in pl.parallel(0, 80, 80):
                group_rows = 80 - group_start
                output_group = pl.slice(a, [group_rows, 128], [group_start, 0])
                direct_capacity = pl.tensor.dim(output_group, 0)
                aliased_output = output_group
                capacity = pl.tensor.dim(aliased_output, 0)
                with pl.at(level=pl.Level.CORE_GROUP):
                    tile: pl.Tile[[80, 128], pl.FP32] = pl.load(
                        output_group, [capacity - direct_capacity, 0], [80, 128]
                    )
                    pl.store(tile, [0, 0], c)
            return c

        @pl.function
        def main(
            self,
            a: pl.Tensor[[80, 128], pl.FP32],
            c: pl.InOut[pl.Tensor[[80, 128], pl.FP32]],
        ) -> pl.Tensor[[80, 128], pl.FP32]:
            self.c1_group(a, c)
            return c

    result = _legalize_outlined(Before)
    assert "capacity" not in _scalar_param_names(_graph_func(result, "c1_group"))


def test_num_tokens_derived_view_extent_is_rejected():
    """`graph_rebind_tensor` patches a view's address, never its shape.

    The recorded template's `shapes`/`strides` are replayed as-is, so an
    extent read from a boundary scalar would apply call one's shape to a
    later call's buffer.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.Graph)
        def layer(
            self,
            w: pl.Tensor[[512, 128], pl.FP32],
            c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            num_tokens: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            wl: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.slice(w, [num_tokens, 128], [0, 0])
            with pl.at(level=pl.Level.CORE_GROUP):
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(wl, [0, 0], [128, 128])
                pl.store(t, [0, 0], c)
            return c

        @pl.function
        def main(
            self,
            w: pl.Tensor[[512, 128], pl.FP32],
            c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            c = self.layer(w, c, 128)
            return c

    with pytest.raises(ValueError, match="shape is not a compile-time constant"):
        _legalize_outlined(Before)


def test_num_tokens_derived_view_offset_is_rejected():
    """The case that would genuinely freeze.

    `num_tokens + i * 128` can be neither hoisted (`i` does not exist at the
    call site) nor frozen (`num_tokens` is patched per call), so the recorded
    delta would be call one's while the address is call two's.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.Graph)
        def tiled(
            self,
            w: pl.Tensor[[512, 256], pl.FP32],
            c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
            num_tokens: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            for i in pl.range(2):
                off = num_tokens + i * 128
                band: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.slice(w, [128, 128], [off, 0])
                with pl.at(level=pl.Level.CORE_GROUP):
                    t: pl.Tile[[128, 128], pl.FP32] = pl.load(band, [0, 0], [128, 128])
                    pl.store(t, [0, 0], c)
            return c

        @pl.function
        def main(
            self,
            w: pl.Tensor[[512, 256], pl.FP32],
            c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            self.tiled(w, c, 0)
            return c

    with pytest.raises(ValueError, match="nor the same on every replay"):
        _legalize_outlined(Before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
