# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for the TensorType.manual_dep flag.

The flag tells the runtime to skip OverlapMap dependency tracking for the
buffer; the user is responsible for ordering. Phase 1 wiring covers
construction, structural equality, hashing, printing, and the DSL surface
(``pl.Tensor[..., pl.ManualDep]``).
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir


class TestManualDepConstruction:
    """Constructor accepts manual_dep; default is False."""

    def test_default_is_false(self):
        t = ir.TensorType([16, 32], DataType.FP32)
        assert t.manual_dep is False

    def test_kwarg_true(self):
        t = ir.TensorType([16, 32], DataType.FP32, manual_dep=True)
        assert t.manual_dep is True

    def test_with_memref_and_manual_dep(self):
        span = ir.Span.unknown()
        memref = ir.MemRef(
            ir.MemorySpace.DDR,
            ir.ConstInt(0x1000, DataType.INT64, span),
            16 * 32 * 4,
            42,
        )
        t = ir.TensorType([16, 32], DataType.FP32, memref=memref, manual_dep=True)
        assert t.manual_dep is True
        assert t.memref is not None

    def test_with_tensor_view_and_manual_dep(self):
        tv = ir.TensorView([], ir.TensorLayout.NZ)
        t = ir.TensorType([16, 32], DataType.FP32, tensor_view=tv, manual_dep=True)
        assert t.manual_dep is True
        assert t.tensor_view is not None


class TestStructuralEquality:
    """Two TensorTypes that differ only in manual_dep are not structurally equal."""

    def test_equal_when_both_default(self):
        t1 = ir.TensorType([16, 32], DataType.FP32)
        t2 = ir.TensorType([16, 32], DataType.FP32)
        ir.assert_structural_equal(t1, t2)  # raises if mismatched

    def test_equal_when_both_true(self):
        t1 = ir.TensorType([16, 32], DataType.FP32, manual_dep=True)
        t2 = ir.TensorType([16, 32], DataType.FP32, manual_dep=True)
        ir.assert_structural_equal(t1, t2)

    def test_unequal_when_only_one_marked(self):
        t1 = ir.TensorType([16, 32], DataType.FP32, manual_dep=True)
        t2 = ir.TensorType([16, 32], DataType.FP32, manual_dep=False)
        with pytest.raises(Exception, match="manual_dep"):
            ir.assert_structural_equal(t1, t2)


class TestStructuralHash:
    """Different manual_dep produces different structural hashes."""

    def test_distinct_hashes(self):
        t1 = ir.TensorType([16, 32], DataType.FP32, manual_dep=True)
        t2 = ir.TensorType([16, 32], DataType.FP32, manual_dep=False)
        assert ir.structural_hash(t1) != ir.structural_hash(t2)


class TestDSLSurface:
    """pl.Tensor[..., pl.ManualDep] threads to TensorType.manual_dep."""

    def test_annotation_marker(self):
        T = pl.Tensor[[16, 32], pl.FP32, pl.ManualDep]
        assert T.manual_dep is True

    def test_no_marker_default_false(self):
        T = pl.Tensor[[16, 32], pl.FP32]
        assert T.manual_dep is False

    def test_marker_with_layout(self):
        T = pl.Tensor[[16, 32], pl.FP32, pl.NZ, pl.ManualDep]
        assert T.manual_dep is True
        assert T.layout == ir.TensorLayout.NZ

    def test_function_param_propagates(self):
        @pl.program
        class P:
            @pl.function
            def f(self, x: pl.Tensor[[16, 32], pl.FP32, pl.ManualDep]):
                return x

        func = list(P.functions.values())[0]
        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert param_type.manual_dep is True

    def test_function_param_default_false(self):
        @pl.program
        class P:
            @pl.function
            def f(self, x: pl.Tensor[[16, 32], pl.FP32]):
                return x

        func = list(P.functions.values())[0]
        param_type = func.params[0].type
        assert isinstance(param_type, ir.TensorType)
        assert param_type.manual_dep is False


class TestPrinterRoundTrip:
    """Printer emits pl.ManualDep marker and parser strips it back."""

    def test_print_includes_marker(self):
        @pl.program
        class P:
            @pl.function
            def f(self, x: pl.Tensor[[16, 32], pl.FP32, pl.ManualDep]):
                return x

        printed = P.as_python()
        assert "pl.ManualDep" in printed

    def test_print_omits_marker_when_default(self):
        @pl.program
        class P:
            @pl.function
            def f(self, x: pl.Tensor[[16, 32], pl.FP32]):
                return x

        printed = P.as_python()
        assert "pl.ManualDep" not in printed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
