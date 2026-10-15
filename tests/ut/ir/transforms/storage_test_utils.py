# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Comparison of storage bindings in hand-written legacy pass expectations."""

from pypto import ir
from pypto.pypto_core.ir import _without_type_memref


def without_call_memrefs(program: ir.Program) -> ir.Program:
    """Remove inferred Call result allocations, retaining all variable storage.

    Storage passes allocate variables, while DSL operator inference can copy
    input MemRefs into Call result types. These tests specify the variables'
    storage contract, not those redundant inferred copies. Exact comparison
    still checks every variable's allocation identity, offset, size and slots,
    as well as every other Call type field. Roundtrip instruments compare the
    original IR without this normalization.
    """

    class NormalizeCalls(ir.IRMutator):
        def visit_call(self, op: ir.Call) -> ir.Expr:
            call = super().visit_call(op)
            assert isinstance(call, ir.Call)
            return ir.Call(
                call.op, call.args, call.kwargs, call.attrs, _without_type_memref(call.type), call.span
            )

    return NormalizeCalls().visit_program(program)
