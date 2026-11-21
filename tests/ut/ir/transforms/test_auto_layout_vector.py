# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check logical element and axis mappings after native layout selection."""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.debug import torch_codegen

_TILE_RESHAPE = ir.get_op("tile.reshape").name
_TILE_SELS = ir.get_op("tile.sels").name


def _select(program: ir.Program) -> ir.Program:
    prefix = passes.PassPipeline()
    for step in ir.PassManager.get_strategy(ir.OptimizationStrategy.AutoLayout).passes:
        prefix.add_pass(step)
        if step.get_name() == "AutoLayout":
            return prefix.run(program)
    raise AssertionError("AutoLayout missing from pipeline")


def test_each_store_uses_its_own_destination_layout(ascend_backend):
    @pl.program
    class Stores:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[32, 32], pl.FP32],
            out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            dn: pl.Out[
                pl.Tensor[[32, 32], pl.FP32, pl.TensorView(stride=[1, 32], layout=pl.TensorLayout.DN)]
            ],
        ):
            data = pl.load(x, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            mask = pl.tile.tri(0, [32, 32], dtype=pl.FP32)
            out = pl.store(mask, [0, 0], out)
            dn = pl.store(data, [0, 0], dn)

    body = next(iter(_select(Stores).functions.values())).body
    assert isinstance(body, ir.SeqStmts)
    layouts = []
    for stmt in body.stmts:
        if isinstance(stmt, (ir.AssignStmt, ir.EvalStmt)):
            call = stmt.value if isinstance(stmt, ir.AssignStmt) else stmt.expr
            if isinstance(call, ir.Call) and call.op.name == ir.get_op("tile.store").name:
                tile = call.args[0].type
                assert isinstance(tile, ir.TileType)
                layouts.append(tile.get_effective_tile_view().blayout)
    assert layouts == [ir.TileLayout.row_major, ir.TileLayout.col_major]


def test_shape_changing_reshape_preserves_flat_element_order(ascend_backend):
    @pl.program
    class Reshape:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            x: pl.Tensor[[32, 32], pl.FP32, pl.TensorView(stride=[1, 32], layout=pl.TensorLayout.DN)],
            out: pl.Out[
                pl.Tensor[[16, 64], pl.FP32, pl.TensorView(stride=[1, 16], layout=pl.TensorLayout.DN)]
            ],
        ) -> pl.Tensor[[16, 64], pl.FP32, pl.TensorView(stride=[1, 16], layout=pl.TensorLayout.DN)]:
            tile = pl.load(x, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            result = pl.tile.reshape(tile, [16, 64])
            return pl.store(result, [0, 0], out)

    body = next(iter(_select(Reshape).functions.values())).body
    assert isinstance(body, ir.SeqStmts)
    call = next(
        s.value
        for s in body.stmts
        if isinstance(s, ir.AssignStmt) and isinstance(s.value, ir.Call) and s.value.op.name == _TILE_RESHAPE
    )
    src = call.args[0].type
    dst = call.type
    assert isinstance(src, ir.TileType) and isinstance(dst, ir.TileType)
    src_col = src.get_effective_tile_view().blayout == ir.TileLayout.col_major
    dst_col = dst.get_effective_tile_view().blayout == ir.TileLayout.col_major
    for row in range(16):
        for col in range(64):
            offset = col * 16 + row if dst_col else row * 64 + col
            input_value = (offset % 32) * 32 + offset // 32 if src_col else offset
            assert input_value == row * 64 + col


def test_two_row_reductions_preserve_logical_axis(tmp_path, ascend_backend):
    @pl.program
    class Reduce:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 1], pl.FP32]]
        ) -> pl.Tensor[[32, 1], pl.FP32]:
            tile = pl.load(x, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            scratch = pl.tile.create([32, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            row = pl.tile.row_sum(tile, scratch)
            scratch2 = pl.tile.create([32, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            twice = pl.tile.row_sum(row, scratch2)
            return pl.store(twice, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self, x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 1], pl.FP32]]
        ) -> pl.Tensor[[32, 1], pl.FP32]:
            return self.kernel(x, out)

    ir.compile(
        Reduce,
        strategy=ir.OptimizationStrategy.AutoLayout,
        backend_type=backend.BackendType.Ascend910,
        skip_ptoas=True,
        output_dir=str(tmp_path),
    )
    pto = next(tmp_path.rglob("*.pto")).read_text()
    store = next(line for line in pto.splitlines() if "pto.tstore" in line)
    assert "rows=32, cols=1," in store

    torch = pytest.importorskip("torch")
    scope = {}
    exec(torch_codegen(_select(Reduce), check_shapes=True), scope)
    data = torch.arange(1024, dtype=torch.float32).reshape(32, 32)
    output = torch.full((32, 1), -777.0)
    scope["main"](data, output)
    torch.testing.assert_close(output, data.sum(dim=1, keepdim=True))


def test_sels_keeps_packed_predicate_geometry(ascend_backend):
    @pl.program
    class Select:
        @pl.function(type=pl.FunctionType.InCore)
        def producer(
            self,
            x: pl.Tensor[[16, 16], pl.FP16],
            w: pl.Tensor[[16, 16], pl.FP16],
            dst: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            a = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            b = pl.load(w, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            return pl.store(pl.matmul(a, b), [0, 0], dst)

        @pl.function(type=pl.FunctionType.InCore)
        def select(
            self, src: pl.Tensor[[16, 16], pl.FP32], dst: pl.Out[pl.Tensor[[16, 16], pl.FP32]]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            data = pl.load(src, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
            mask = pl.tile.full([16, 16], dtype=pl.INT32, value=1)
            scratch = pl.tile.full([16, 16], dtype=pl.FP32, value=0.0)
            result = pl.tile.sels(mask, data, scratch, -1.0)
            return pl.store(result, [0, 0], dst)

        @pl.function(type=pl.FunctionType.InCore)
        def consume(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            w: pl.Tensor[[16, 16], pl.FP32],
            dst: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            a = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            b = pl.load(w, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            return pl.store(pl.matmul(a, b), [0, 0], dst)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[16, 16], pl.FP16],
            y: pl.Tensor[[16, 16], pl.FP16],
            w: pl.Tensor[[16, 16], pl.FP32],
            dst: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            t = pl.tensor.create([16, 16], dtype=pl.FP32)
            u = pl.tensor.create([16, 16], dtype=pl.FP32)
            t = self.producer(x, y, t)
            u = self.select(t, u)
            return self.consume(u, w, dst)

    kernel = _select(Select).get_function("select")
    assert kernel is not None
    body = kernel.body
    assert isinstance(body, ir.SeqStmts)
    selected = next(
        s.value
        for s in body.stmts
        if isinstance(s, ir.AssignStmt) and isinstance(s.value, ir.Call) and s.value.op.name == _TILE_SELS
    )
    # The constant 1 selects the first column of each 32-element group per logical row.
    # Treating each INT32 word as an ordinary NZ element changes which bits apply.
    for operand in selected.args[:2]:
        tile = operand.type
        assert isinstance(tile, ir.TileType)
        assert [dim.value for dim in tile.shape if isinstance(dim, ir.ConstInt)] == [16, 16]


@pytest.mark.parametrize("to_dn", [True, False])
def test_rectangular_matrix_conversion_permutes_storage(tmp_path, ascend_backend, to_dn):
    dense = "pl.Tensor[[32,64], pl.FP32]"
    column = "pl.Tensor[[32,64], pl.FP32, pl.TensorView(stride=[1,32], layout=pl.TensorLayout.DN)]"
    source, target = (dense, column) if to_dn else (column, dense)
    program = pl.parse(
        f"""
@pl.program
class CopyMatrix:
    @pl.function(type=pl.FunctionType.InCore)
    def copy(self, x: {source}, out: pl.Out[{target}]) -> {target}:
        value = pl.load(x, [0,0], [32,64], target_memory=pl.Mem.Vec)
        return pl.store(value, [0,0], out)
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: {source}, out: pl.Out[{target}]) -> {target}:
        return self.copy(x, out)
""",
        filename=__file__,
    )
    assert isinstance(program, ir.Program)
    ir.compile(
        program,
        strategy=ir.OptimizationStrategy.AutoLayout,
        backend_type=backend.BackendType.Ascend910,
        skip_ptoas=True,
        output_dir=str(tmp_path),
    )
    # Vec TMOV copies corresponding physical bytes. Changing a flat matrix's
    # major axis therefore needs a physical permutation in the generated kernel.
    pto = next(tmp_path.rglob("*.pto")).read_text()
    assert "pto.ttrans " in pto
    torch = pytest.importorskip("torch")
    data = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64)
    output = torch.full_like(data, -1)
    namespace = {}
    exec(torch_codegen(_select(program), check_shapes=True), namespace)
    namespace["main"](data, output)
    torch.testing.assert_close(output, data)


@pytest.mark.parametrize(
    ("mutation", "source_dn"),
    [
        ("none", False),
        ("direct", False),
        ("branch", False),
        ("write_result", False),
        ("direct", True),
        ("branch", True),
    ],
)
def test_vec_conversion_reads_the_current_backing(tmp_path, ascend_backend, mutation, source_dn):
    nd = "pl.Tensor[[32,64], pl.FP32]"
    dn = "pl.Tensor[[32,64], pl.FP32, pl.TensorView(stride=[1,32], layout=pl.TensorLayout.DN)]"
    source = dn if source_dn else nd
    alias = (
        "pl.tile.write(a, [0,0], pl.const(0.0, pl.FP32))"
        if mutation == "write_result"
        else "pl.tile.reshape(a, [32,64])"
    )
    observed = "alias" if mutation == "write_result" else "a"
    update = {
        "none": "        changed = alias",
        "direct": "        changed = pl.tile.assemble(a, patch_tile, [8, 0])",
        "write_result": "        changed = pl.tile.assemble(a, patch_tile, [8, 0])",
        "branch": """        if replace:
            changed = pl.tile.assemble(alias, patch_tile, [8, 0])
        else:
            changed = alias""",
    }[mutation]
    program = pl.parse(
        f"""
@pl.program
class BackingReads:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: {source},
        patch: pl.Tensor[[8,64], pl.FP32],
        replace: pl.Scalar[pl.BOOL],
        before: pl.Out[{dn}],
        after: pl.Out[{dn}],
        final: pl.Out[pl.Tensor[[32,64], pl.FP32]],
    ) -> pl.Tensor[[32,64], pl.FP32]:
        a = pl.load(x, [0,0], [32,64], target_memory=pl.Mem.Vec)
        alias = {alias}
        patch_tile = pl.load(patch, [0,0], [8,64], target_memory=pl.Mem.Vec)
        before = pl.store({observed}, [0,0], before)
{update}
        after = pl.store({observed}, [0,0], after)
        return pl.store(changed, [0,0], final)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: {source},
        patch: pl.Tensor[[8,64], pl.FP32],
        replace: pl.Scalar[pl.BOOL],
        before: pl.Out[{dn}],
        after: pl.Out[{dn}],
        final: pl.Out[pl.Tensor[[32,64], pl.FP32]],
    ) -> pl.Tensor[[32,64], pl.FP32]:
        return self.kernel(x, patch, replace, before, after, final)
""",
        filename=__file__,
    )
    assert isinstance(program, ir.Program)
    ir.compile(
        program,
        strategy=ir.OptimizationStrategy.AutoLayout,
        backend_type=backend.BackendType.Ascend910,
        skip_ptoas=True,
        output_dir=str(tmp_path),
    )
    pto = next(tmp_path.rglob("*.pto")).read_text()
    instructions = [
        "transpose" if "pto.ttrans " in line else "store"
        for line in pto.splitlines()
        if "pto.ttrans " in line or "pto.tstore " in line
    ]
    # The fixed DN stores share one transpose only when the backing is immutable.
    expected = ["transpose", "store", "store", "store"]
    if mutation != "none":
        expected.insert(2, "transpose")
    if source_dn:
        expected.insert(0, "transpose")  # Initialize the row-major mutable backing from a DN load.
    assert instructions == expected

    torch = pytest.importorskip("torch")
    namespace = {}
    exec(torch_codegen(_select(program), check_shapes=True), namespace)
    data = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64)
    if source_dn:
        data = data.T.contiguous().T
    patch = torch.full((8, 64), -42.0)
    for replace in [True, False] if mutation == "branch" else [True]:
        before = torch.empty_strided((32, 64), (1, 32), dtype=torch.float32)
        after = torch.empty_like(before)
        final = torch.empty_like(data)
        namespace["main"](data, patch, replace, before, after, final)
        expected_data = data.clone()
        if mutation in {"direct", "write_result"} or (mutation == "branch" and replace):
            expected_data[8:16] = patch
        torch.testing.assert_close(before, data)
        torch.testing.assert_close(after, expected_data)
        torch.testing.assert_close(final, expected_data)
