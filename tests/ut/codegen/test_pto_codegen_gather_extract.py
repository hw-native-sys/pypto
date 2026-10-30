# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 axis gather must repack column slices before using flat indices."""

import pypto.language as pl
import pytest
from pypto import codegen, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def _codegen_a5(program):
    # The tensor.gather printer currently emits (source, idx, dim=-1),
    # which cannot roundtrip through the DSL's positional dim argument.
    # Keep property verification while testing the full codegen pipeline.
    verifier = passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)
    with passes.PassContext([verifier]):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    kernel = next(f for f in optimized.functions.values() if f.func_type == ir.FunctionType.AIV)
    return codegen.PTOCodegen().generate(ir.Program([kernel], kernel.name, optimized.span))


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("dtype", [pl.FP32, pl.FP16, pl.INT32, pl.INT16])
@pytest.mark.parametrize("strided", [False, True])
def test_a5_gather_column_slice_uses_extract(dtype, strided, ascend_backend):
    # Sixteen rows cross the A5 FP32 vector box boundary. For a column slice,
    # the parent pitch is 128 but tgather's flat indices assume a pitch of 64.
    if strided:

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                inp: pl.Tensor[[16, 128], dtype],
                idx: pl.Tensor[[16, 64], pl.INT32],
                output: pl.Out[pl.Tensor[[16, 64], dtype]],
            ) -> pl.Tensor[[16, 64], dtype]:
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
                idx: pl.Tensor[[16, 64], pl.INT32],
                output: pl.Out[pl.Tensor[[16, 64], dtype]],
            ) -> pl.Tensor[[16, 64], dtype]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.tensor.gather(inp, dim=-1, index=idx)
                    output = pl.assemble(output, out, [0, 0])
                return output

    mlir = _codegen_a5(Program)
    # A5 TEXTRACT cannot read INT16/INT32. Those sources keep their
    # existing assembly path; float gather must eliminate source subviews.
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
        assert "rows=16, cols=64" in extracts[0]
        assert ", %c0_index, %c64_index :" in extracts[0]
        packed = extracts[0].split("outs(")[1].split(" :")[0]
        assert f"pto.tgather ins({packed}," in mlir
    else:
        # GM input is already packed by tload. Integer local input must
        # not acquire a TEXTRACT unsupported by the A5 verifier.
        assert not extracts


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("dtype, aligned_cols", [(pl.FP32, 8), (pl.FP16, 16)])
@pytest.mark.parametrize("case", ["source", "destination", "both", "aligned"])
@pytest.mark.parametrize("index_rows", [8, 16])
def test_a5_gather_checks_parent_and_destination_pitch(dtype, aligned_cols, case, index_rows, ascend_backend):
    parent_cols = 2 * aligned_cols + (case in ("source", "both"))
    source_cols = aligned_cols + (case in ("destination", "both"))

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Opaque)
        def main(
            self,
            inp: pl.Tensor[[16, parent_cols], dtype],
            idx: pl.Tensor[[index_rows, 64], pl.INT32],
            output: pl.Out[pl.Tensor[[index_rows, 64], dtype]],
        ) -> pl.Tensor[[index_rows, 64], dtype]:
            with pl.at(level=pl.Level.CORE_GROUP):
                local = pl.create_tensor([16, parent_cols], dtype=dtype)
                local = pl.assemble(local, inp, [0, 0])
                source = local[:, :source_cols]
                alias = source
                window = alias[:, :source_cols]
                shaped = pl.reshape(window, [16, source_cols])
                valid = pl.tensor.set_validshape(shaped, 16, source_cols)
                out = pl.tensor.gather(valid, dim=-1, index=idx)
                output = pl.assemble(output, out, [0, 0])
            return output

    # A slice/alias/reshape/validity chain must retain the parent's pitch.
    # In the source-only case the slice itself has an aligned column count,
    # so checking just its TileType would incorrectly choose TEXTRACT.
    # These unaligned IR cases check fallback selection; PTOAS separately
    # rejects their unaligned physical allocations even without TEXTRACT.
    mlir = _codegen_a5(Program)
    extracts = [line for line in mlir.splitlines() if "pto.textract ins(" in line]
    assert len(extracts) == (case == "aligned")
    assert mlir.count("pto.tgather ins(") == 1
    if case != "aligned":
        # A smaller index leading dimension must bound the fallback source;
        # an unbounded 16-row view cannot be assembled into an 8-row tile.
        copies = [line for line in mlir.splitlines() if "pto.tinsert ins(" in line]
        assert any(f"rows={index_rows}, cols={source_cols}" in line for line in copies)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
