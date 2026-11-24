# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Parser tests for ``pld.tensor.put_async`` / ``get_async``.

The async URMA ops return ``dst`` (the result aliases argument 0) and carry a
compile-time ``engine`` kwarg. The printer restores the enum spelling
``pld.CommEngine.URMA`` so a printed program re-parses to the same IR without
losing the engine or the destination alias.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import Error, passes
from pypto.pypto_core import ir


def _get_func(program: ir.Program, name: str) -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None
    return program.functions[gvar]


def _iter_stmts(stmt: ir.Stmt):
    """Yield ``stmt`` and every nested statement (flattening ``SeqStmts``)."""
    yield stmt
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            yield from _iter_stmts(s)
    elif isinstance(stmt, ir.ForStmt):
        yield from _iter_stmts(stmt.body)


def _find_async_call(program: ir.Program, op_name: str) -> ir.Call:
    func = _get_func(program, "kernel")
    matches = [
        stmt.value
        for stmt in _iter_stmts(func.body)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == ir.get_op(op_name).name
    ]
    assert len(matches) == 1, f"expected exactly one {op_name} call, got {len(matches)}"
    return matches[0]


def test_pld_exports_comm_engine():
    """The contract spelling ``pld.CommEngine.URMA`` resolves to the IR enum."""
    assert pld.CommEngine.URMA == ir.CommEngine.URMA
    assert {"put_async", "get_async"} <= set(pld.tensor.__all__)


def test_put_async_round_trips_engine_and_alias():
    """Printed ``put_async`` re-parses with its engine and dst alias intact."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[16, 64], pl.FP16],
            src: pld.DistributedTensor[[16, 64], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            _out = pld.tensor.put_async(dst, peer=peer, src=src, engine=pld.CommEngine.URMA)

    call = _find_async_call(P, "pld.tensor.put_async")
    assert call.kwargs["engine"] == int(ir.CommEngine.URMA)
    # The result aliases dst: its type is the destination's type.
    func = _get_func(P, "kernel")
    ir.assert_structural_equal(call.type, func.params[0].type)

    printed = str(P)
    assert "pld.tensor.put_async" in printed
    assert "pld.CommEngine.URMA" in printed

    reparsed = pl.parse_program(printed)
    ir.assert_structural_equal(P, reparsed)


def test_get_async_round_trips_engine():
    """Printed ``get_async`` re-parses with its engine and dst alias intact."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[16, 64], pl.FP16],
            src: pld.DistributedTensor[[16, 64], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            _out = pld.tensor.get_async(dst, peer=peer, src=src, engine=pld.CommEngine.URMA)

    call = _find_async_call(P, "pld.tensor.get_async")
    assert call.kwargs["engine"] == int(ir.CommEngine.URMA)
    func = _get_func(P, "kernel")
    ir.assert_structural_equal(call.type, func.params[0].type)

    printed = str(P)
    assert "pld.tensor.get_async" in printed
    assert "pld.CommEngine.URMA" in printed

    reparsed = pl.parse_program(printed)
    ir.assert_structural_equal(P, reparsed)


def _verify_through_ssa(program: ir.Program) -> None:
    pipeline = passes.PassPipeline()
    pipeline.add_pass(passes.convert_to_ssa())
    with passes.PassContext([], passes.VerificationLevel.BASIC):
        pipeline.run(program)


@pytest.mark.parametrize(
    ("op", "dst_type", "submit", "setup", "completion"),
    [
        ("pld.tensor.get_async_event", "pl.Tensor", "dst, 1, src", "", "event"),
        ("pld.tile.get_async", "pl.Tensor", "dst, 1, src, session", "src", "event, session"),
        ("pld.tile.put_async", "pld.DistributedTensor", "dst, 1, src, session", "dst", "event, session"),
    ],
)
def test_async_transfer_unpacks_one_submission_and_aliases_dst(
    ascend_backend, default_pass_manager, op, dst_type, submit, setup, completion
):
    make_session = f"session = pld.system.make_session(pld.system.get_comm_ctx({setup}))" if setup else ""
    program = pl.parse_program(f"""
@pl.program
class P:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, dst: {dst_type}[[32], pl.FP32], src: pld.DistributedTensor[[32], pl.FP32]
    ) -> {dst_type}[[32], pl.FP32]:
        {make_session}
        out, event = {op}({submit})
        pld.system.defer_wait_event({completion})
        return out
""")
    call = _find_async_call(program, op)
    assert isinstance(call.type, ir.TupleType)
    ir.assert_structural_equal(call.type.types[0], _get_func(program, "kernel").params[0].type)
    assert isinstance(call.type.types[1], ir.AsyncEventType)
    assert ir.returned_param_indices(_get_func(program, "kernel"), program) == [0]
    assert str(program).count(f"{op}(") == 1
    _verify_through_ssa(program)
    ir.assert_structural_equal(program, pl.parse_program(str(program)))
    ir.assert_structural_equal(program, ir.deserialize(ir.serialize(program)))

    kernel = _get_func(default_pass_manager.run_passes(program), "kernel")
    (out,) = [
        stmt.var
        for stmt in _iter_stmts(kernel.body)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.TupleGetItemExpr)
        and stmt.value.index == 0
    ]
    dst_type = kernel.params[0].type
    assert isinstance(dst_type, ir.TensorType) and dst_type.memref is not None
    assert isinstance(out.type, ir.TensorType) and out.type.memref is dst_type.memref


@pytest.mark.parametrize("op", ["pld.tensor.get_async_event", "pld.tile.get_async", "pld.tile.put_async"])
def test_async_window_result_keeps_destination_context(ascend_backend, default_pass_manager, op):
    window = "dst" if op == "pld.tile.put_async" else "src"
    setup = f"session = pld.system.make_session(pld.system.get_comm_ctx({window}))"
    session_arg = ", session" if op.startswith("pld.tile") else ""
    program = pl.parse_program(f"""
@pl.program
class P:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, dst: pld.DistributedTensor[[32], pl.FP32], src: pld.DistributedTensor[[32], pl.FP32]):
        {setup if session_arg else ""}
        out, event = {op}(dst, 1, src{session_arg})
        pld.system.wait_event(event{session_arg})
        result_ctx = pld.system.get_comm_ctx(out)
        next_session = pld.system.make_session(result_ctx)
        _out, sent = pld.tile.put_async(out, 2, src, next_session)
        pld.system.defer_wait_event(sent)
""")
    kernel = _get_func(default_pass_manager.run_passes(program), "kernel")
    ctx = next(
        stmt
        for stmt in _iter_stmts(kernel.body)
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint.startswith("result_ctx")
    )
    assert ctx.value is kernel.params[-2]  # dst_ctx, followed by src_ctx


@pytest.mark.parametrize("op", ["pld.tensor.get_async_event", "pld.tile.get_async", "pld.tile.put_async"])
@pytest.mark.parametrize("stage", [1, 2])
def test_pipeline_keeps_memory_access_after_wait(ascend_backend, default_pass_manager, op, stage):
    put = op == "pld.tile.put_async"
    dst, src = ("window", "local") if put else ("local", "window")
    session_arg = ", session" if op.startswith("pld.tile.") else ""
    setup = "session = pld.system.make_session(pld.system.get_comm_ctx(window))" if session_arg else ""
    # GET consumes the new destination. PUT can overwrite its source buffer after completion.
    read, write = ("sink", "local") if put else ("out", "sink")
    program = pl.parse_program(f"""
@pl.program
class P:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, local: pl.Tensor[[16, 64], pl.FP32], window: pld.DistributedTensor[[16, 64], pl.FP32],
               sink: pl.Tensor[[16, 64], pl.FP32]):
        {setup}
        for i in pl.pipeline(0, 4, 1, stage={stage}):
            out, event = {op}({dst}, i + 1, {src}{session_arg})
            pld.system.wait_event(event{session_arg})
            data = pl.tile.load({read}, [0, 0], [16, 64])
            {write} = pl.tile.store(data, [0, 0], {write})
""")
    kernel = _get_func(default_pass_manager.run_passes(program), "kernel")
    wait = "pld.system.wait_event"
    access = "tile.store" if put else "tile.load"
    ordered = []
    for stmt in _iter_stmts(kernel.body):
        if isinstance(stmt, ir.AssignStmt):
            expr = stmt.value
        elif isinstance(stmt, ir.EvalStmt):
            expr = stmt.expr
        else:
            continue
        if isinstance(expr, ir.Call) and expr.op.name in (op, wait, access):
            ordered.append(expr.op.name)
    assert ordered == [op, wait, access] * stage


def test_tile_get_event_reuses_session_for_dynamic_peers(ascend_backend):
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[32], pl.FP32],
            src: pld.DistributedTensor[[32], pl.FP32],
            first: pl.Scalar[pl.INT32],
            second: pl.Scalar[pl.INT32],
        ):
            ctx = pld.system.get_comm_ctx(src)
            session = pld.system.make_session(ctx, engine=pld.CommEngine.URMA)
            next_dst, e1 = pld.tile.get_async(dst, first, src, session)
            _dst_after, e2 = pld.tile.get_async(next_dst, second, src, session)
            pld.system.defer_wait_event(e1, session)
            pld.system.defer_wait_event(e2)

    printed = str(P)
    assert printed.count("pld.system.make_session(") == 1
    assert printed.count("pld.tile.get_async(") == 2
    assert "pld.CommEngine.URMA" in printed
    _verify_through_ssa(P)
    ir.assert_structural_equal(P, pl.parse_program(printed))


def test_put_and_get_share_session_across_peers_and_completion_paths(ascend_backend):
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            local: pl.Tensor[[32], pl.FP32],
            window: pld.DistributedTensor[[32], pl.FP32],
            first: pl.Scalar[pl.INT32],
            second: pl.Scalar[pl.INT32],
        ):
            ctx = pld.system.get_comm_ctx(window)
            session = pld.system.make_session(ctx)
            local_after, fetched = pld.tile.get_async(local, first, window, session)
            _ok = pld.system.wait_event(fetched)
            _dst_after, sent = pld.tile.put_async(window, second, local_after, session)
            pld.system.defer_wait_event(sent, session)

    printed = str(P)
    assert printed.count("pld.system.make_session(") == 1
    assert ir.get_op_arg_effect("pld.tile.put_async", 0) == ir.ArgEffect.ReadWrite
    assert ir.get_op_arg_effect("pld.tile.put_async", 2) == ir.ArgEffect.Read
    assert ir.get_op_arg_effect("pld.tile.put_async", 3) == ir.ArgEffect.ReadWrite
    _verify_through_ssa(P)
    ir.assert_structural_equal(P, pl.parse_program(printed))
    ir.assert_structural_equal(P, ir.deserialize(ir.serialize(P)))


def test_reassigned_event_and_session_use_the_reaching_submission(ascend_backend):
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, local: pl.Tensor[[32], pl.FP32], window: pld.DistributedTensor[[32], pl.FP32]):
            ctx = pld.system.get_comm_ctx(window)
            session = pld.system.make_session(ctx)
            local_after, event = pld.tile.get_async(local, 1, window, session)
            pld.system.wait_event(event, session)
            session = pld.system.make_session(ctx)
            _dst_after, event = pld.tile.put_async(window, 2, local_after, session)
            pld.system.defer_wait_event(event, session)

    # The parser reuses each Var on reassignment. Each wait has one reaching
    # definition once ConvertToSSA splits the two session/event pairs.
    _verify_through_ssa(P)


def test_wait_event_on_a_tensor_get_infers_its_session(ascend_backend):
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, dst: pl.Tensor[[32], pl.FP32], src: pld.DistributedTensor[[32], pl.FP32]):
            _out, fetched = pld.tensor.get_async_event(dst, 1, src)
            _ok = pld.system.wait_event(fetched)

    _verify_through_ssa(P)
    ir.assert_structural_equal(P, pl.parse_program(str(P)))


def test_wait_event_rejects_another_session_for_put(ascend_backend):
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, dst: pld.DistributedTensor[[32], pl.FP32], src: pl.Tensor[[32], pl.FP32]):
            ctx = pld.system.get_comm_ctx(dst)
            session = pld.system.make_session(ctx)
            other = pld.system.make_session(ctx)
            _dst_after, event = pld.tile.put_async(dst, 1, src, session)
            pld.system.wait_event(event, other)

    with pytest.raises(Error, match="(?s)after 'ConvertToSSA'.*same session"):
        _verify_through_ssa(P)


def test_defer_wait_event_rejects_a_reassigned_session(ascend_backend):
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, dst: pl.Tensor[[32], pl.FP32], src: pld.DistributedTensor[[32], pl.FP32]):
            ctx = pld.system.get_comm_ctx(src)
            session = pld.system.make_session(ctx)
            _dst_after, event = pld.tile.get_async(dst, 1, src, session)
            session = pld.system.make_session(ctx)
            pld.system.defer_wait_event(event, session)

    with pytest.raises(Error, match="(?s)after 'ConvertToSSA'.*same session"):
        _verify_through_ssa(P)


def test_prefetch_handles_cannot_be_used_for_urma_get_or_defer(ascend_backend):
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, dst: pl.Tensor[[32], pl.FP32], src: pld.DistributedTensor[[32], pl.FP32]):
            ctx = pl.prefetch.make_context()
            session = pl.prefetch.session(ctx)
            event = pl.prefetch.async_prefetch(dst, ctx)
            pld.tile.get_async(dst, 1, src, session)
            pld.system.defer_wait_event(event)

    with pytest.raises(Error, match="after 'ConvertToSSA'") as exc:
        _verify_through_ssa(P)
    assert "session from pld.system.make_session" in str(exc.value)
    assert "distributed transfer event" in str(exc.value)


@pytest.mark.parametrize("completion", ["wait_event", "defer_wait_event"])
def test_tensor_get_event_rejects_explicit_session(ascend_backend, completion):
    program = pl.parse_program(f"""
@pl.program
class P:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, dst: pl.Tensor[[32], pl.FP32], src: pld.DistributedTensor[[32], pl.FP32]):
        ctx = pld.system.get_comm_ctx(src)
        session = pld.system.make_session(ctx)
        out, event = pld.tensor.get_async_event(dst, 1, src)
        pld.system.{completion}(event, session)
""")
    with pytest.raises(Error, match=rf"{completion}.*omit the session"):
        _verify_through_ssa(program)


@pytest.mark.parametrize("condition", ["cond", "pl.const(1, pl.INDEX) == pl.const(1, pl.INDEX)"])
def test_prefetch_wait_accepts_prefetch_events_from_either_branch(
    ascend_backend, default_pass_manager, condition
):
    program = pl.parse_program(f"""
@pl.program
class P:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[32], pl.FP32], y: pl.Tensor[[32], pl.FP32], cond: pl.Scalar[pl.BOOL]):
        ctx = pl.prefetch.make_context()
        session = pl.prefetch.session(ctx)
        if {condition}:
            event = pl.prefetch.async_prefetch(x, ctx)
        else:
            event = pl.prefetch.async_prefetch(y, ctx)
        pl.prefetch.wait(event, session)
""")
    assert "pl.prefetch.wait(" in str(default_pass_manager.run_passes(program))


def test_prefetch_wait_accepts_loop_carried_prefetch_event(ascend_backend):
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[32], pl.FP32], count: pl.Scalar[pl.INDEX], refresh: pl.Scalar[pl.BOOL]
        ):
            ctx = pl.prefetch.make_context()
            session = pl.prefetch.session(ctx)
            event = pl.prefetch.async_prefetch(x, ctx)
            for _ in pl.range(count):
                if refresh:
                    event = pl.prefetch.async_prefetch(x, ctx)
            pl.prefetch.wait(event, session)

    _verify_through_ssa(P)


def test_prefetch_wait_rejects_unknown_event_after_branch(ascend_backend):
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pl.Tensor[[32], pl.FP32],
            src: pld.DistributedTensor[[32], pl.FP32],
            cond: pl.Scalar[pl.BOOL],
        ):
            pc = pl.prefetch.make_context()
            ps = pl.prefetch.session(pc)
            if cond:
                event = pl.prefetch.async_prefetch(dst, pc)
            else:
                ctx = pld.system.get_comm_ctx(src)
                session = pld.system.make_session(ctx)
                _dst_after, event = pld.tile.get_async(dst, 1, src, session)
            pl.prefetch.wait(event, ps)

    with pytest.raises(Error, match="prefetch.wait requires an event traceable to prefetch.async_prefetch"):
        _verify_through_ssa(P)


def _subregion_program(op_name: str):
    """Build a one-kernel program using the requested async op with a subregion."""

    if op_name == "put_async":

        @pl.program
        class PutProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                dst: pld.DistributedTensor[[16, 64], pl.FP16],
                src: pld.DistributedTensor[[16, 64], pl.FP16],
                peer: pl.Scalar[pl.INT32],
            ):
                _out = pld.tensor.put_async(
                    dst,
                    peer=peer,
                    src=src,
                    dst_offsets=[0, 0],
                    src_offsets=[0, 0],
                    shape=[1, 32],
                    engine=pld.CommEngine.URMA,
                )

        return PutProgram

    @pl.program
    class GetProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[16, 64], pl.FP16],
            src: pld.DistributedTensor[[16, 64], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            _out = pld.tensor.get_async(
                dst,
                peer=peer,
                src=src,
                dst_offsets=[0, 0],
                src_offsets=[0, 0],
                shape=[1, 32],
                engine=pld.CommEngine.URMA,
            )

    return GetProgram


@pytest.mark.parametrize("op_name", ["put_async", "get_async"])
def test_async_subregion_round_trips(op_name):
    """The subregion form (offsets + shape) prints and reparses losslessly."""
    program = _subregion_program(op_name)
    call = _find_async_call(program, f"pld.tensor.{op_name}")
    assert call.kwargs["engine"] == int(ir.CommEngine.URMA)
    printed = str(program)
    assert "pld.CommEngine.URMA" in printed
    reparsed = pl.parse_program(printed)
    ir.assert_structural_equal(program, reparsed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
