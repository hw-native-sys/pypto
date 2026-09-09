# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Persistence and identity of the final function representation boundary."""

import pytest
from pypto import DataType, ir

SPAN = ir.Span.unknown()


def _function(stage=ir.FunctionIRStage.Functional, requires_runtime_binding=False):
    value = ir.Var("value", ir.ScalarType(DataType.INDEX), SPAN)
    body = ir.AssignStmt(value, ir.ConstInt(1, DataType.INDEX, SPAN), SPAN)
    return ir.Function(
        "kernel",
        [],
        [],
        body,
        SPAN,
        type=ir.FunctionType.InCore,
        requires_runtime_binding=requires_runtime_binding,
        ir_stage=stage,
    )


def test_function_stage_defaults_and_constructor_are_independent_of_execution_type():
    assert set(ir.FunctionIRStage.__members__) == {"Functional", "Buffer"}
    assert _function().ir_stage == ir.FunctionIRStage.Functional
    kernel = _function(ir.FunctionIRStage.Buffer)
    assert kernel.ir_stage == ir.FunctionIRStage.Buffer
    assert kernel.func_type == ir.FunctionType.InCore
    assert kernel.level == ir.Level.CHIP_DIE
    with pytest.raises(AttributeError):
        setattr(kernel, "ir_stage", ir.FunctionIRStage.Functional)


@pytest.mark.parametrize("stage", list(ir.FunctionIRStage))
def test_function_builder_preserves_stage(stage):
    builder = ir.IRBuilder()
    with builder.function("kernel", span=SPAN, type=ir.FunctionType.InCore, ir_stage=stage) as function:
        pass
    assert function.get_result().ir_stage == stage


def test_function_stage_participates_in_structural_identity():
    functional = _function()
    buffer = _function(ir.FunctionIRStage.Buffer)
    assert not ir.structural_equal(functional, buffer, enable_auto_mapping=True)
    assert ir.structural_hash(functional) != ir.structural_hash(buffer)
    with pytest.raises(ValueError, match="FunctionIRStage mismatch"):
        ir.assert_structural_equal(functional, buffer, enable_auto_mapping=True)


@pytest.mark.parametrize("stage", list(ir.FunctionIRStage))
def test_complete_program_round_trip_preserves_function_stage(stage):
    kernel = _function(stage)
    wrapper = ir.Function("wrapper", [], [], ir.SeqStmts([], SPAN), SPAN, type=ir.FunctionType.Group)
    program = ir.Program([kernel, wrapper], "Stages", SPAN)
    restored = ir.deserialize(ir.serialize(program))
    assert isinstance(restored, ir.Program)
    restored_kernel = restored.get_function("kernel")
    restored_wrapper = restored.get_function("wrapper")
    assert restored_kernel is not None and restored_kernel.ir_stage == stage
    assert restored_wrapper is not None and restored_wrapper.ir_stage == ir.FunctionIRStage.Functional
    ir.assert_structural_equal(program, restored, enable_auto_mapping=True)
    assert ir.structural_hash(program) == ir.structural_hash(restored)


def _serialized_buffer_function():
    payload = ir.serialize(_function(ir.FunctionIRStage.Buffer))
    field = b"\xa8ir_stage\x01"
    assert payload.count(field) == 1
    return payload, field


def test_serialized_function_without_stage_defaults_to_functional():
    payload, field = _serialized_buffer_function()
    # Rename the entry to an unknown field, retaining a well-formed MessagePack
    # map while reproducing the absence of ir_stage in pre-boundary blobs.
    restored = ir.deserialize(payload.replace(field, b"\xa8old_hint\x01"))
    assert isinstance(restored, ir.Function)
    assert restored.ir_stage == ir.FunctionIRStage.Functional


@pytest.mark.parametrize("encoded", [b"\x02", b"\xff", b"\xc0", b"\xc2", b"\xa1x", b"\x90"])
def test_serialized_function_rejects_invalid_present_stage(encoded):
    payload, field = _serialized_buffer_function()
    with pytest.raises(ValueError, match="Invalid FunctionIRStage"):
        ir.deserialize(payload.replace(field, b"\xa8ir_stage" + encoded))


@pytest.mark.parametrize("stage", list(ir.FunctionIRStage))
@pytest.mark.parametrize("runtime_binding", [False, True])
def test_generic_mutator_rebuild_preserves_stage_and_runtime_binding(stage, runtime_binding):
    class ReplaceConstant(ir.IRMutator):
        def visit_const_int(self, constant):
            return ir.ConstInt(2, constant.dtype, constant.span)

    original = _function(stage, runtime_binding)
    changed = ReplaceConstant().visit_function(original)
    assert changed is not original
    assert isinstance(changed.body, ir.AssignStmt)
    assert isinstance(changed.body.value, ir.ConstInt)
    assert changed.body.value.value == 2
    assert changed.ir_stage == stage
    assert changed.requires_runtime_binding == runtime_binding
    assert changed.func_type == original.func_type
    assert changed.level == original.level
    assert changed.role == original.role


def test_buffer_printer_marker_is_diagnostic_and_functional_output_stays_unchanged():
    assert "ir_stage" not in _function().as_python()
    printed = _function(ir.FunctionIRStage.Buffer).as_python()
    assert "# ir_stage: Buffer\n" in printed
    assert "ir_stage=" not in printed
