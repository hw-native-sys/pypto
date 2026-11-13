# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unlowered logical synchronization must never disappear in device codegen."""

import pytest
from pypto import DataType, codegen, ir
from pypto.ir.op import sync
from pypto.pypto_core import ir as core_ir

pytestmark = pytest.mark.usefixtures("ascend_backend")


@pytest.mark.parametrize(
    "operation", ["record", "completed", "join", "memory_overlaps", "wait", "vf_barrier"]
)
@pytest.mark.parametrize("inside_vf", [False, True])
def test_codegen_requires_logical_sync_lowering(operation: str, inside_vf: bool):
    """Reject before helper outlining or a no-result emitter can drop a wait."""
    span = ir.Span.unknown()
    if operation == "record":
        call = sync.record(ir.PipeType.V)
    elif operation == "completed":
        call = sync.completed(ir.PipeType.V)
    elif operation in {"join", "wait"}:
        # A token parameter isolates the wait boundary from its defining op.
        # The check must reject the wait itself, before checking the kernel ABI.
        token = ir.Var("token", ir.SyncTokenType(ir.PipeType.V), span)
        call = sync.join(token, token) if operation == "join" else sync.wait(token, ir.PipeType.MTE3)
    elif operation == "memory_overlaps":
        handle = ir.Var("storage", ir.TensorType([16], DataType.FP32), span)
        call = sync.memory_overlaps(
            handle, handle, first_begin=-64, first_end=0, second_begin=0, second_end=64
        )
    else:
        call = sync.vf_barrier()

    body = ir.EvalStmt(call, span)
    if inside_vf:
        body = ir.RegScopeStmt("vf", body, span)
    function = ir.Function("kernel", [], [], body, span, type=ir.FunctionType.AIV)
    program = ir.Program([function], "LogicalSyncBoundary", span)
    with pytest.raises(ValueError, match="Logical synchronization requires LowerSync"):
        codegen.PTOCodegen().generate(program)


@pytest.mark.parametrize("boundary", ["parameter", "return"])
@pytest.mark.parametrize("nested", [False, True])
def test_codegen_rejects_token_abi_without_sync_calls(boundary: str, nested: bool):
    """A missing record cannot make an unrepresentable device ABI acceptable."""
    span = ir.Span.unknown()
    type_ = ir.SyncTokenType(ir.PipeType.MTE2)
    if nested:
        type_ = ir.TupleType([type_])
    parameters = [ir.Var("incoming", type_, span)] if boundary == "parameter" else []
    returns = [type_] if boundary == "return" else []
    function = ir.Function(
        "kernel", parameters, returns, ir.SeqStmts([], span), span, type=ir.FunctionType.AIV
    )
    program = ir.Program([function], "TokenAbiBoundary", span)
    with pytest.raises(ValueError, match=f"SyncToken cannot cross a function {boundary} boundary"):
        codegen.PTOCodegen().generate(program)


def test_codegen_rejects_orphan_token_without_sync_calls():
    """Raw manually built tokens still require a local completion definition."""
    span = ir.Span.unknown()
    orphan = ir.Var("orphan", ir.SyncTokenType(ir.PipeType.MTE2), span)
    function = ir.Function("kernel", [], [], ir.EvalStmt(orphan, span), span, type=ir.FunctionType.AIV)
    program = ir.Program([function], "OrphanTokenBoundary", span)
    with pytest.raises(ValueError, match="must be defined before use"):
        codegen.PTOCodegen().generate(program)


def test_codegen_rejects_logical_operation_encoded_as_task_launch():
    """A raw Submit must not hide an unlowered operation in its callee field."""
    span = ir.Span.unknown()
    submit = ir.Submit(ir.get_op("sync.vf_barrier"), [], [], ir.ScalarType(DataType.TASK_ID), span)
    function = ir.Function("kernel", [], [], ir.EvalStmt(submit, span), span, type=ir.FunctionType.AIV)
    program = ir.Program([function], "LogicalSubmitBoundary", span)
    with pytest.raises(ValueError, match="Logical synchronization requires LowerSync"):
        codegen.PTOCodegen().generate(program)


@pytest.mark.parametrize("field", ["base", "size"])
@pytest.mark.parametrize("location", ["parameter", "return"])
def test_codegen_rejects_token_in_window_signature_metadata_without_sync_calls(field, location):
    """A valid distributed ABI must not conceal token-valued window metadata."""
    span = ir.Span.unknown()
    orphan = ir.Var("orphan", ir.SyncTokenType(ir.PipeType.MTE2), span)
    pointer = ir.Var("window_base", ir.PtrType(), span)
    size = ir.ConstInt(2048, DataType.INDEX, span)
    malformed_window = ir.WindowBuffer(
        orphan if field == "base" else pointer,
        orphan if field == "size" else size,
        span=span,
    )
    shape = [ir.ConstInt(value, DataType.INDEX, span) for value in (16, 32)]
    malformed = ir.DistributedTensorType(shape, DataType.FP32, malformed_window)
    ordinary = ir.DistributedTensorType(shape, DataType.FP32, ir.WindowBuffer(pointer, size, span=span))
    incoming = ir.Var("incoming", malformed if location == "parameter" else ordinary, span)
    context = ir.Var("context", ir.CommCtxType(), span)
    # The companion CommCtx is mandatory for a DistributedTensor parameter;
    # include it so the intended metadata error cannot be an ABI pairing error.
    function = ir.Function(
        "kernel",
        [incoming, context],
        [malformed] if location == "return" else [],
        ir.SeqStmts([], span),
        span,
        type=ir.FunctionType.AIV,
    )
    program = ir.Program([function], "WindowMetadataBoundary", span)
    with pytest.raises(ValueError, match="metadata"):
        codegen.PTOCodegen().generate(program)


def test_codegen_accepts_ordinary_window_metadata_with_a_companion_context():
    """Rejecting hidden tokens must preserve the ordinary distributed parameter ABI."""
    span = ir.Span.unknown()
    pointer = ir.Var("window_base", ir.PtrType(), span)
    window = ir.WindowBuffer(pointer, ir.ConstInt(2048, DataType.INDEX, span), span=span)
    shape = [ir.ConstInt(value, DataType.INDEX, span) for value in (16, 32)]
    incoming = ir.Var("incoming", ir.DistributedTensorType(shape, DataType.FP32, window), span)
    context = ir.Var("context", ir.CommCtxType(), span)
    function = ir.Function(
        "kernel", [incoming, context], [], ir.SeqStmts([], span), span, type=ir.FunctionType.AIV
    )
    program = ir.Program([function], "OrdinaryWindowBoundary", span)
    assert "@kernel(" in codegen.PTOCodegen().generate(program)


@pytest.mark.parametrize(
    "location",
    [
        "call_kwarg",
        "submit_kwarg",
        "function_attr",
        "loop_attr",
        "scope_attr",
        "runtime_scope_attr",
        "type_geometry",
        "window_size",
    ],
)
def test_codegen_finds_pure_logical_predicates_inside_metadata(location):
    """BOOL-valued logical ops also need lowering when hidden in metadata.

    These expressions contain no token, so checking token ownership alone
    cannot enforce the unlowered-codegen boundary.
    """
    span = ir.Span.unknown()
    storage = ir.Var("storage", ir.TensorType([16], DataType.FP32), span)
    predicate = sync.memory_overlaps(
        storage, storage, first_begin=0, first_end=64, second_begin=0, second_end=64
    )
    parameters: list[ir.Var] = [storage]
    body: ir.Stmt = ir.SeqStmts([], span)
    attributes: dict[str, object] = {}
    if location == "call_kwarg":
        # Explicit construction permits Expr-valued physical-call kwargs. A
        # preflight that only walks ordinary operands would miss this predicate.
        call = ir.Call(ir.Op("system.bar_all"), [], {"hidden": predicate}, ir.VoidType(), span)
        body = ir.EvalStmt(call, span)
    elif location == "submit_kwarg":
        submit = ir.Submit(
            ir.GlobalVar("callee"),
            [],
            [],
            {"hidden": predicate},
            None,
            ir.ScalarType(DataType.TASK_ID),
            span,
        )
        body = ir.EvalStmt(submit, span)
    elif location == "function_attr":
        attributes["hidden"] = predicate
    elif location == "loop_attr":
        body = ir.ForStmt(
            ir.Var("i", ir.ScalarType(DataType.INDEX), span),
            ir.ConstInt(0, DataType.INDEX, span),
            ir.ConstInt(1, DataType.INDEX, span),
            ir.ConstInt(1, DataType.INDEX, span),
            [],
            body,
            [],
            span,
            attrs={"hidden": predicate},
        )
    elif location == "scope_attr":
        body = ir.RegScopeStmt("vf", body, span, attrs={"hidden": predicate})
    elif location == "runtime_scope_attr":
        # RuntimeScope has no attrs argument on its direct Python constructor.
        builder = core_ir.IRBuilder()
        builder.begin_function("builder", span, type=ir.FunctionType.AIV)
        builder.begin_scope(ir.ScopeKind.Runtime, span, manual=False, attrs=[("hidden", predicate)])
        body = builder.end_scope(span)
        builder.end_function(span)
    elif location == "type_geometry":
        parameters.append(ir.Var("view", ir.TensorType([predicate], DataType.FP32), span))
    else:
        window = ir.WindowBuffer(ir.Var("window_base", ir.PtrType(), span), predicate, span=span)
        shape = [ir.ConstInt(value, DataType.INDEX, span) for value in (16, 32)]
        parameters.extend(
            [
                ir.Var("view", ir.DistributedTensorType(shape, DataType.FP32, window), span),
                ir.Var("context", ir.CommCtxType(), span),
            ]
        )
    function = ir.Function("kernel", parameters, [], body, span, type=ir.FunctionType.AIV, attrs=attributes)
    program = ir.Program([function], "HiddenLogicalPredicate", span)
    with pytest.raises(ValueError, match="Logical synchronization requires LowerSync"):
        codegen.PTOCodegen().generate(program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
