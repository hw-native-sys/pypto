# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Axis gather preserves source addressing across Ascend backends."""

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def _codegen(program):
    # The tensor.gather printer currently emits (source, idx, dim=-1),
    # which cannot roundtrip through the DSL's positional dim argument.
    # Keep property verification while testing the full codegen pipeline.
    verifier = passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)
    with passes.PassContext([verifier]):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    kernel = next(f for f in optimized.functions.values() if f.func_type == ir.FunctionType.AIV)
    return codegen.PTOCodegen().generate(ir.Program([kernel], kernel.name, optimized.span))


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910, BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("dtype", [pl.FP32, pl.FP16, pl.INT32, pl.INT16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("index_rows", [8, 16])
def test_gather_column_slice_uses_parent_pitch(dtype, strided, index_rows, ascend_backend):
    # Sixteen rows cross the A5 FP32 vector box boundary. Gather from the
    # original parent using r*128 + 64 + idx, without packing its narrow view.
    if strided:

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                inp: pl.Tensor[[16, 128], dtype],
                idx: pl.Tensor[[index_rows, 64], pl.INT32],
                output: pl.Out[pl.Tensor[[index_rows, 64], dtype]],
            ) -> pl.Tensor[[index_rows, 64], dtype]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    local = pl.create_tensor([16, 128], dtype=dtype)
                    local = pl.assemble(local, inp, [0, 0])
                    source = local[:, 64:128]
                    out = pl.tensor.gather(source, dim=-1, index=idx)
                    output = pl.assemble(output, out, [0, 0])
                return output

    else:

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                inp: pl.Tensor[[16, 64], dtype],
                idx: pl.Tensor[[index_rows, 64], pl.INT32],
                output: pl.Out[pl.Tensor[[index_rows, 64], dtype]],
            ) -> pl.Tensor[[index_rows, 64], dtype]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.tensor.gather(inp, dim=-1, index=idx)
                    output = pl.assemble(output, out, [0, 0])
                return output

    mlir = _codegen(Program)
    assert "pto.textract" not in mlir
    assert mlir.count("pto.tgather ins(") == 1
    if strided:
        gather = next(line for line in mlir.splitlines() if "pto.tgather ins(" in line)
        assert "rows=16, cols=128" in gather
        assert f"rows={index_rows}, cols=64" in gather
        assert any("pto.tmuls ins(" in line and "%c128_i32" in line for line in mlir.splitlines())
        assert any("pto.tadds ins(" in line and "%c64_i32" in line for line in mlir.splitlines())
        assert not any("pto.tadds ins(" in line and "%c0_i" in line for line in mlir.splitlines())


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910, BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("rows", [1, 16])
def test_gather_nested_window_keeps_row_and_column_origin(ascend_backend, rows):
    # The nested window begins at parent[6,64], so element offsets must be
    # r*256 + 1600 + idx. In particular a one-row gather still needs the origin.
    program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.Opaque)
    def main(self, idx: pl.Tensor[[{rows},64],pl.INT32],
             output: pl.Out[pl.Tensor[[{rows},64],pl.FP32]]) -> pl.Tensor[[{rows},64],pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            parent = pl.full([32,256],dtype=pl.FP32,value=1.0)
            outer = parent[4:28,32:224]
            window = outer[2:18,32:96]
            alias = window
            picked = pl.tensor.gather(alias,dim=-1,index=idx)
            output = pl.assemble(output,picked,[0,0])
        return output
""")
    mlir = _codegen(program)
    gather = next(line for line in mlir.splitlines() if "pto.tgather ins(" in line)
    assert "rows=32, cols=256" in gather
    assert "pto.textract" not in mlir
    assert any("pto.tadds ins(" in line and "%c1600_i32" in line for line in mlir.splitlines())
    if rows > 1:
        assert any("pto.tmuls ins(" in line and "%c256_i32" in line for line in mlir.splitlines())


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("rows,parent_cols,needs_copy", [(16, 256, False), (64, 1024, True)])
def test_gather_int16_offsets_use_parent_only_when_representable(
    ascend_backend, rows, parent_cols, needs_copy
):
    # The logical 64x64 window fits signed INT16 offsets, but its last row in
    # the 64x1024 parent does not. Retain packing rather than narrowing offsets.
    # A5 supports the INT16/INT32 index casts used by this lowering.
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Opaque)
        def main(
            self,
            idx: pl.Tensor[[rows, 16], pl.INT16],
            output: pl.Out[pl.Tensor[[rows, 16], pl.FP16]],
        ) -> pl.Tensor[[rows, 16], pl.FP16]:
            with pl.at(level=pl.Level.CORE_GROUP):
                parent = pl.full([rows, parent_cols], dtype=pl.FP16, value=1.0)
                window = parent[:, 64:128]
                picked = pl.tensor.gather(window, dim=-1, index=idx)
                output = pl.assemble(output, picked, [0, 0])
            return output

    mlir = _codegen(Program)
    assert ("pto.textract" in mlir) == needs_copy
    gather = next(line for line in mlir.splitlines() if "pto.tgather ins(" in line)
    expected_cols = 64 if needs_copy else parent_cols
    assert f"rows={rows}, cols={expected_cols}" in gather


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910, BackendType.Ascend950], indirect=True)
def test_rank3_gather_uses_one_full_tile(ascend_backend):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Opaque)
        def main(
            self,
            inp: pl.Tensor[[2, 8, 64], pl.FP32],
            idx: pl.Tensor[[2, 8, 32], pl.INT32],
            output: pl.Out[pl.Tensor[[2, 8, 32], pl.FP32]],
        ) -> pl.Tensor[[2, 8, 32], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                out = pl.tensor.gather(inp, dim=-1, index=idx)
                output = pl.assemble(output, out, [0, 0, 0])
            return output

    mlir = _codegen(Program)
    assert mlir.count("pto.tgather ins(") == 1
    assert "scf.for" not in mlir
    assert "rows=16, cols=64" in mlir
    assert "rows=16, cols=32" in mlir


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910, BackendType.Ascend950], indirect=True)
def test_gather_reuses_packed_full_slice_reshape(ascend_backend):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Opaque)
        def main(
            self,
            inp: pl.Tensor[[32, 512], pl.FP32],
            idx: pl.Tensor[[1, 2048], pl.INT32],
            output: pl.Out[pl.Tensor[[1, 2048], pl.FP32]],
        ) -> pl.Tensor[[1, 2048], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                local = pl.add(inp, 1.0)
                full = local[:, :]
                flat = pl.reshape(full, [1, 16384])
                out = pl.tensor.gather(flat, dim=-1, index=idx)
                output = pl.assemble(output, out, [0, 0])
            return output

    mlir = _codegen(Program)
    assert "pto.textract" not in mlir
    assert mlir.count("pto.tgather ins(") == 1


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910, BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("rank3", [False, True])
def test_large_gm_gather_uses_bounded_source_rows(ascend_backend, rank3):
    source_shape = [2, 32, 4096] if rank3 else [64, 4096]
    index_shape = [2, 32, 8] if rank3 else [64, 8]
    output_offsets = [0, 0, 0] if rank3 else [0, 0]

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Opaque)
        def main(
            self,
            inp: pl.Tensor[source_shape, pl.FP32],
            idx: pl.Tensor[index_shape, pl.INT32],
            output: pl.Out[pl.Tensor[index_shape, pl.FP32]],
        ) -> pl.Tensor[index_shape, pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                out = pl.tensor.gather(inp, dim=-1, index=idx)
                output = pl.assemble(output, out, output_offsets)
            return output

    mlir = _codegen(Program)
    assert "scf.for" in mlir
    assert mlir.count("pto.tgather ins(") == 1
    assert "rows=1, cols=4096" in mlir
    assert "rows=64, cols=4096" not in mlir


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910, BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize(
    "source_shape,index_shape",
    [([64, 512], [64, 128]), ([32, 512], [32, 768]), ([2, 16, 512], [2, 8, 1600])],
)
def test_large_local_gather_is_memory_bounded(ascend_backend, source_shape, index_shape):
    output_offsets = [0] * len(index_shape)

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Opaque)
        def main(
            self,
            idx: pl.Tensor[index_shape, pl.INT32],
            output: pl.Out[pl.Tensor[index_shape, pl.FP32]],
        ) -> pl.Tensor[index_shape, pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                local = pl.full(source_shape, dtype=pl.FP32, value=1.0)
                out = pl.tensor.gather(local, dim=-1, index=idx)
                output = pl.assemble(output, out, output_offsets)
            return output

    mlir = _codegen(Program)
    assert "scf.for" in mlir
    assert mlir.count("pto.tgather ins(") == 1
    if len(source_shape) == 3:
        # Source and index middle extents differ; preserve all 32 source rows
        # and map each output row through its original ND coordinates.
        assert "rows=32, cols=512" in mlir
        assert "arith.muli" in mlir


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910, BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("case", ["prefix", "source_view", "index_view"])
def test_gather_counts_backing_storage_and_window_copy(ascend_backend, case):
    if case == "prefix":

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                idx: pl.Tensor[[32, 8], pl.INT32],
                output: pl.Out[pl.Tensor[[32, 8], pl.FP32]],
            ) -> pl.Tensor[[32, 8], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    local = pl.full([64, 512], dtype=pl.FP32, value=1.0)
                    out = pl.tensor.gather(local, dim=-1, index=idx)
                    output = pl.assemble(output, out, [0, 0])
                return output

    elif case == "source_view":

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                idx: pl.Tensor[[64, 8], pl.INT32],
                output: pl.Out[pl.Tensor[[64, 8], pl.FP32]],
            ) -> pl.Tensor[[64, 8], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    parent = pl.full([64, 640], dtype=pl.FP32, value=1.0)
                    window = parent[:, :128]
                    out = pl.tensor.gather(window, dim=-1, index=idx)
                    output = pl.assemble(output, out, [0, 0])
                return output

    else:

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                inp: pl.Tensor[[64, 128], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 8], pl.FP32]],
            ) -> pl.Tensor[[64, 8], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    parent = pl.full([64, 640], dtype=pl.INT32, value=0)
                    window = parent[:, :8]
                    out = pl.tensor.gather(inp, dim=-1, index=window)
                    output = pl.assemble(output, out, [0, 0])
                return output

    mlir = _codegen(Program)
    capacity = backend.get_backend_instance(ascend_backend).get_mem_size(ir.MemorySpace.Vec)
    # A known source parent is addressed directly, eliminating the old window
    # copy. A large index backing still coexists with the GM source load.
    assert ("scf.for" in mlir) == (case == "index_view" and capacity < 192 * 1024)
    assert mlir.count("pto.tgather ins(") == 1


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910, BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize(
    "dtype_name,source_cols", [("FP32", 744), ("FP16", 1488), ("INT32", 744), ("INT16", 1488)]
)
@pytest.mark.parametrize("local_index", [False, True])
def test_rank3_gather_reuses_full_packed_local_source(ascend_backend, dtype_name, source_cols, local_index):
    # Every dtype has a 95,232-byte packed source with 32-byte-aligned rows.
    # The source and gather temporaries fit A2/A3 Vec, but a second full source
    # does not. A new reshape SSA value must not lose the original packing proof.
    index_init = "indices = pl.full([2, 16, 32], dtype=pl.INT32, value=0)" if local_index else "indices = idx"
    program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        idx: pl.Tensor[[2, 16, 32], pl.INT32],
        output: pl.Out[pl.Tensor[[2, 16, 32], pl.{dtype_name}]],
    ) -> pl.Tensor[[2, 16, 32], pl.{dtype_name}]:
        with pl.at(level=pl.Level.CORE_GROUP):
            local = pl.full([2, 16, {source_cols}], dtype=pl.{dtype_name}, value=1)
            {index_init}
            picked = pl.tensor.gather(local, dim=-1, index=indices)
            output = pl.assemble(output, picked, [0, 0, 0])
        return output
""")

    mlir = _codegen(program)
    assert "scf.for" not in mlir
    assert "pto.subview" not in mlir
    assert "pto.textract" not in mlir
    assert "pto.tadds" not in mlir
    gathers = [line for line in mlir.splitlines() if "pto.tgather ins(" in line]
    assert len(gathers) == 1
    assert f"rows=32, cols={source_cols}" in gathers[0]


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910, BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize(
    "source_shape,dtype", [([1, 3], pl.FP32), ([4, 3], pl.FP32), ([4, 3], pl.FP16), ([2, 2, 3], pl.FP32)]
)
def test_gather_unaligned_source_rows_use_row_fallback(ascend_backend, source_shape, dtype):
    index_shape = [*source_shape[:-1], 16]
    offsets = [0] * len(source_shape)

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Opaque)
        def main(
            self,
            inp: pl.Tensor[source_shape, dtype],
            idx: pl.Tensor[index_shape, pl.INT32],
            output: pl.Out[pl.Tensor[index_shape, dtype]],
        ) -> pl.Tensor[index_shape, dtype]:
            with pl.at(level=pl.Level.CORE_GROUP):
                out = pl.tensor.gather(inp, dim=-1, index=idx)
                output = pl.assemble(output, out, offsets)
            return output

    mlir = _codegen(Program)
    if source_shape != [1, 3]:
        assert "scf.for" in mlir
    assert mlir.count("pto.tgather ins(") == 1
    # Only the physical allocation is padded; GM reads retain three valid columns.
    allocs = [line for line in mlir.splitlines() if "pto.alloc_tile" in line]
    padded_cols = 16 if dtype == pl.FP16 else 8
    assert any(f"cols={padded_cols}" in line and "valid_col = %c3_index" in line for line in allocs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
