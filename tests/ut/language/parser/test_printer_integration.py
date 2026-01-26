# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration tests for parser and printer round-trip."""

import pypto
import pypto.language as pl
from pypto.pypto_core import DataType, ir


class TestPrinterIntegration:
    """Tests for printer integration with new subscript syntax."""

    def test_tensor_type_printed_with_subscript(self):
        """Test that TensorType is printed with subscript notation."""
        tensor_type = ir.TensorType([64, 128], DataType.FP16)

        result = pypto.ir.python_print(tensor_type)

        # Should use subscript notation
        assert "pl.Tensor[[64, 128], pl.FP16]" in result
        # Should NOT use call notation
        assert "pl.Tensor((" not in result

    def test_tile_type_printed_with_subscript(self):
        """Test that TileType is printed with subscript notation."""
        tile_type = ir.TileType([16, 16], DataType.FP32)

        result = pypto.ir.python_print(tile_type)

        # Should use subscript notation
        assert "pl.Tile[[16, 16], pl.FP32]" in result
        # Should NOT use call notation
        assert "pl.Tile((" not in result

    def test_function_printed_with_subscript_types(self):
        """Test that function parameters use subscript notation."""

        @pl.function
        def test_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.op.tensor.cast(x, target_type=pl.FP32)
            return result

        # Print the function
        printed = pypto.ir.python_print(test_func)

        # Check subscript notation is used
        assert "pl.Tensor[[64, 128], pl.FP16]" in printed
        assert "pl.Tensor[[64, 128], pl.FP32]" in printed
        # Check old notation is NOT used
        assert "pl.Tensor((" not in printed

    def test_parsed_function_printer_round_trip(self):
        """Test that parsed functions can be printed correctly."""

        @pl.function
        def round_trip(
            x: pl.Tensor[[64], pl.FP32],
            y: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            sum_val: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, y)
            result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.mul(sum_val, 2.0)
            return result

        # Print and check syntax
        printed = pypto.ir.python_print(round_trip)

        assert "def round_trip" in printed
        assert "pl.Tensor[[64], pl.FP32]" in printed
        # Printer uses simplified tensor operation notation
        assert "tensor.add" in printed or "pl.op.tensor.add" in printed
