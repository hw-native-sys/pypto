# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Registry contracts for the internal PTO target operations."""

import pytest
from pypto.pypto_core import ir


@pytest.mark.parametrize(
    "op_name", ["pto.alloc_tile", "pto.tload", "pto.tsqrt", "pto.tadd", "pto.tmul", "pto.tstore"]
)
def test_minimal_pto_target_ops_are_registered_and_internal(op_name: str):
    assert ir.is_op_registered(op_name)
    assert ir.get_pto_op_spec(op_name) is not None

    with pytest.raises(Exception, match="internal-only"):
        ir.create_op_call(op_name, [], ir.Span.unknown())


def test_tadd_schema_has_two_read_inputs_and_one_write_output():
    spec = ir.get_pto_op_spec("pto.tadd")

    assert spec is not None
    assert spec["operand_groups"] == [
        {"role": "input", "effect": "read", "type": "tile_buffer", "min_count": 2, "max_count": 2},
        {"role": "output", "effect": "write", "type": "tile_buffer", "min_count": 1, "max_count": 1},
    ]
    assert spec["result_kind"] == "none"
    assert spec["result_effect"] == "none"
    assert not spec["pure"]


def test_tsqrt_schema_has_one_read_input_and_one_write_output():
    spec = ir.get_pto_op_spec("pto.tsqrt")

    assert spec is not None
    assert spec["operand_groups"] == [
        {"role": "input", "effect": "read", "type": "tile_buffer", "min_count": 1, "max_count": 1},
        {"role": "output", "effect": "write", "type": "tile_buffer", "min_count": 1, "max_count": 1},
    ]
    assert spec["result_kind"] == "none"
    assert spec["result_effect"] == "none"
    assert not spec["pure"]


def test_alloc_schema_models_optional_address_and_allocate_result():
    spec = ir.get_pto_op_spec("pto.alloc_tile")

    assert spec is not None
    assert spec["operand_groups"] == [
        {"role": "metadata", "effect": "none", "type": "scalar", "min_count": 0, "max_count": 1},
        {"role": "metadata", "effect": "none", "type": "scalar", "min_count": 2, "max_count": 2},
    ]
    assert spec["result_kind"] == "tile_buffer"
    assert spec["result_effect"] == "allocate"
    assert not spec["pure"]


def test_high_level_ops_have_no_pto_target_schema():
    assert ir.get_pto_op_spec("tile.add") is None
    assert ir.get_pto_op_spec("tensor.add") is None
    assert ir.get_pto_op_spec("does.not.exist") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
