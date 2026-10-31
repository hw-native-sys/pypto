# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Axis gather shares full-tile source packing across Ascend backends."""

import pypto.language as pl
import pytest
from pypto import codegen, ir, passes
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
def test_gather_column_slice_uses_extract(dtype, strided, index_rows, ascend_backend):
    # Sixteen rows cross the A5 FP32 vector box boundary. For a column slice,
    # the parent pitch is 128 but tgather's flat indices assume a pitch of 64.
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
    # TEXTRACT cannot read INT16/INT32; integer sources use exact adds(0).
    subviews = [line for line in mlir.splitlines() if "pto.subview" in line]
    floating = dtype in (pl.FP32, pl.FP16)
    if floating:
        assert not subviews
    assert mlir.count("pto.tgather ins(") == 1
    extracts = [line for line in mlir.splitlines() if "pto.textract ins(" in line]
    if strided and floating:
        assert len(extracts) == 1
        # The nested slice offsets must fold into a read of the wider
        # parent. The output must be dense and feed tgather directly.
        assert "rows=16, cols=128" in extracts[0]
        assert f"rows={index_rows}, cols=64" in extracts[0]
        assert ", %c0_index, %c64_index :" in extracts[0]
        packed = extracts[0].split("outs(")[1].split(" :")[0]
        assert f"pto.tgather ins({packed}," in mlir
    else:
        # GM input is already packed by tload. Integer local input must
        # not acquire a TEXTRACT unsupported by the backend verifier.
        assert not extracts
        if strided:
            adds = [line for line in mlir.splitlines() if "pto.tadds ins(" in line]
            assert len(adds) == 1
            packed = adds[0].split("outs(")[1].split(" :")[0]
            assert f"pto.tgather ins({packed}," in mlir


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
