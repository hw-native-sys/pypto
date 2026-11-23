# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""IR schema and serialization coverage for HOST CCU registration handles."""

import pytest
from pypto import ir


def test_registration_type_and_attr_schema():
    span = ir.Span.unknown()
    registration = ir.Var("reg", ir.CcuRegistrationType.get(), span)
    restored = ir.deserialize(ir.serialize(registration))
    assert isinstance(restored, ir.Var)
    assert isinstance(restored.type, ir.CcuRegistrationType)
    call = ir.Call(ir.GlobalVar("chip"), [], {}, {"ccu_bindings": []}, ir.TupleType([]), span)
    assert call.attrs["ccu_bindings"] == []
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        ir.Call(
            ir.GlobalVar("chip"),
            [],
            {},
            {"ccu_bindings": [ir.Var("x", ir.PtrType.get(), span)]},
            ir.TupleType([]),
            span,
        )


def test_registration_op_is_internal_only():
    span = ir.Span.unknown()
    allocation = ir.Var("allocation", ir.PtrType.get(), span)
    assert ir.get_op("pld.system.register_ccu").name == "pld.system.register_ccu"
    with pytest.raises(ValueError, match="internal-only"):
        ir.create_op_call(
            "pld.system.register_ccu", [allocation], {"callee": "kernel", "chip_callee": "chip"}, span
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
