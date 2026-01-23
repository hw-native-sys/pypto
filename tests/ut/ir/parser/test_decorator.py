# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for @pl.function decorator."""

import pypto
import pypto.language as pl
import pytest
from pypto.ir.parser.diagnostics import ParserTypeError
from pypto.pypto_core import ir


class TestDecorator:
    """Tests for @pl.function decorator."""

    def test_simple_function(self):
        """Test parsing simple function with no control flow."""

        @pl.function
        def add_tensors(
            x: pl.Tensor[[64, 128], pl.FP16],
            y: pl.Tensor[[64, 128], pl.FP16],
        ) -> pl.Tensor[[64, 128], pl.FP16]:
            result: pl.Tensor[[64, 128], pl.FP16] = pl.op.tensor.add(x, y)
            return result

        assert isinstance(add_tensors, ir.Function)
        assert add_tensors.name == "add_tensors"
        assert len(add_tensors.params) == 2
        assert len(add_tensors.return_types) == 1

    def test_function_with_multiple_statements(self):
        """Test function with multiple statements."""

        @pl.function
        def multi_op(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            a: pl.Tensor[[64], pl.FP32] = pl.op.tensor.mul(x, 2.0)
            b: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(a, 1.0)
            c: pl.Tensor[[64], pl.FP32] = pl.op.tensor.sub(b, 0.5)
            return c

        assert isinstance(multi_op, ir.Function)
        assert multi_op.name == "multi_op"

    def test_function_with_multiple_params(self):
        """Test function with multiple parameters."""

        @pl.function
        def three_param(
            x: pl.Tensor[[64], pl.FP32],
            y: pl.Tensor[[64], pl.FP32],
            z: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            temp: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, y)
            result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(temp, z)
            return result

        assert len(three_param.params) == 3

    def test_function_with_tensor_create(self):
        """Test function that creates tensors."""

        @pl.function
        def create_tensor(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[64, 128], pl.FP32]:
            result: pl.Tensor[[64, 128], pl.FP32] = pl.op.tensor.create([64, 128], dtype=pl.FP32)
            return result

        assert isinstance(create_tensor, ir.Function)

    def test_function_with_binary_ops(self):
        """Test function with binary operations."""

        @pl.function
        def binary_ops(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            # Using operator overloading
            result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(
                pl.op.tensor.mul(x, 2.0), pl.op.tensor.create([64], dtype=pl.FP32)
            )
            return result

        assert isinstance(binary_ops, ir.Function)

    def test_function_with_list_arguments(self):
        """Test function that uses list arguments."""

        @pl.function
        def with_lists(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[32, 64], pl.FP32]:
            # view takes list arguments
            result: pl.Tensor[[32, 64], pl.FP32] = pl.op.tensor.view(x, [32, 64], [0, 0])
            return result

        assert isinstance(with_lists, ir.Function)

    def test_function_serialization(self):
        """Test that parsed functions can be serialized."""

        @pl.function
        def simple(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        # Should be able to serialize
        data = pypto.ir.serialize(simple)
        assert len(data) > 0

        # Should be able to deserialize
        restored = pypto.ir.deserialize(data)
        assert isinstance(restored, ir.Function)
        assert restored.name == "simple"

    def test_function_with_different_dtypes(self):
        """Test function with various data types."""

        @pl.function
        def dtypes(
            fp16: pl.Tensor[[64], pl.FP16],
            fp32: pl.Tensor[[64], pl.FP32],
            int32: pl.Tensor[[64], pl.INT32],
        ) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(
                pl.op.tensor.cast(fp16, target_type=pl.FP32), fp32
            )
            return result

        assert len(dtypes.params) == 3

    def test_invalid_function_no_annotations(self):
        """Test that function without annotations raises error."""

        with pytest.raises(ParserTypeError, match="missing type annotation"):

            @pl.function
            def no_annotations(x):
                return x

    def test_function_preserves_name(self):
        """Test that function name is preserved."""

        @pl.function
        def my_custom_function_name(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        assert my_custom_function_name.name == "my_custom_function_name"

    def test_function_with_negative_numbers(self):
        """Test function with negative number literals."""

        @pl.function
        def with_negatives(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, -1.5)
            return result

        assert isinstance(with_negatives, ir.Function)
