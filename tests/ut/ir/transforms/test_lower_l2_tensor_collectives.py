# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Tests for ``LowerL2TensorCollectives``.

The pass turns a managed collective written in a CHIP/L2 orchestration body
into a call to a synthesized AIV kernel whose implementation is the builtin
template source. There is no ``Expected`` ``@pl.program`` to compare against:
the synthesized function has no DSL surface (its identity lives in the
``builtin_template_dir`` / ``builtin_template_vars`` attrs, which the decorator
cannot spell), so these tests pin the lowered shape directly — the callee, its
signature and directions, the argument order, and the absence of any residual
public collective.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.pypto_core import ir, passes

SIZE = 64
NR = 2
MAX_RECV = 4
TOTAL = NR * MAX_RECV

_KERNEL_NAME = "__builtin_all_to_all_v__fp32"
_ALL_TO_ALL_V = ir.get_op("pld.tensor.all_to_all_v").name


def _build_program(core_num: int = 1):
    """A CHIP pipeline that stages, exchanges, then consumes."""

    @pl.program
    class L2AllToAllV:
        @pl.function(type=pl.FunctionType.InCore)
        def stage_step(
            self,
            inp: pl.Tensor[[TOTAL, SIZE], pl.FP32],
            stage: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]:
            for row in pl.range(TOTAL):
                chunk = pl.load(inp, [row, 0], [1, SIZE])
                stage = pl.store(chunk, [row, 0], stage)
            return stage

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            out: pl.Out[pl.Tensor[[TOTAL, SIZE], pl.FP32]],
        ) -> pl.Tensor[[TOTAL, SIZE], pl.FP32]:
            for row in pl.range(TOTAL):
                chunk = pl.load(data, [row, 0], [1, SIZE])
                out = pl.store(chunk, [row, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            inp: pl.Tensor[[TOTAL, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[TOTAL, SIZE], pl.FP32]],
            stage: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            counts: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        ) -> pl.Tensor[[TOTAL, SIZE], pl.FP32]:
            stage = self.stage_step(inp, stage)
            data = pld.tensor.all_to_all_v(stage, data, signal, counts, recv, core_num=core_num)
            return self.consume_step(data, out)

    return L2AllToAllV


def _get_func(program: ir.Program, name: str) -> ir.Function | None:
    gvar = program.get_global_var(name)
    return None if gvar is None else program.functions[gvar]


def _collect_calls(stmt: ir.Stmt) -> list[ir.Call]:
    found: list[ir.Call] = []

    class _Collector(ir.IRVisitor):
        def visit_call(self, op: ir.Call) -> None:
            found.append(op)

    _Collector().visit_stmt(stmt)
    return found


def _callee_names(func: ir.Function) -> list[str]:
    return [c.op.name for c in _collect_calls(func.body)]


def test_collective_becomes_a_local_builtin_kernel_call():
    """The public collective is replaced by a call to the synthesized kernel."""
    result = passes.lower_l2_tensor_collectives()(_build_program())

    pipeline = _get_func(result, "chip_pipeline")
    assert pipeline is not None
    names = _callee_names(pipeline)
    assert _ALL_TO_ALL_V not in names
    assert names == ["stage_step", _KERNEL_NAME, "consume_step"], names


def test_synthesized_kernel_signature_and_directions():
    """The kernel is an AIV function carrying the five collective operands.

    ``core_num`` is the public op's 6th argument but is gated (compile-time 1)
    and dropped by this rail, because a kernel parameter would become a dispatch
    slot ahead of the CommCtx suffix and shift the shared `kernel.cpp.in`'s
    fixed ``args[5]`` CommContext read. The ABI stays exactly the HOST rail's
    entry-built one.
    """
    result = passes.lower_l2_tensor_collectives()(_build_program())

    kernel = _get_func(result, _KERNEL_NAME)
    assert kernel is not None
    assert kernel.func_type == pl.FunctionType.AIV
    assert [p.name_hint for p in kernel.params] == [
        "input",
        "target",
        "signal",
        "send_counts",
        "recv_counts",
    ]
    # Directions are what orders compute -> collective -> consume once
    # DeriveCallDirections and AutoDeriveTaskDependencies run over the call.
    assert kernel.param_directions == [
        ir.ParamDirection.In,
        ir.ParamDirection.InOut,
        ir.ParamDirection.InOut,
        ir.ParamDirection.In,
        ir.ParamDirection.InOut,
    ]


def test_synthesized_kernel_carries_builtin_template_attrs():
    """The kernel names its source indirectly, for the backend to render."""
    result = passes.lower_l2_tensor_collectives()(_build_program())

    kernel = _get_func(result, _KERNEL_NAME)
    assert kernel is not None
    attrs = dict(kernel.attrs)
    assert attrs["builtin_template_dir"] == ":pypto.runtime.builtins.collectives.all_to_all_v"
    # dtype is the only substitution either rail makes: both reach the same
    # argument layout, so the rendered kernel source is byte-identical between
    # them (asserted end-to-end in the ST).
    template_vars = dict(item.split("=", 1) for item in attrs["builtin_template_vars"].split(","))
    assert template_vars == {"dtype_cpp": "float"}


def test_synthesized_kernel_returns_its_target_parameter():
    """The kernel returns `target` so the call site stays a plain rebind."""
    result = passes.lower_l2_tensor_collectives()(_build_program())

    kernel = _get_func(result, _KERNEL_NAME)
    assert kernel is not None
    assert isinstance(kernel.body, ir.ReturnStmt)
    assert len(kernel.body.value) == 1
    assert kernel.body.value[0] is kernel.params[1]


def test_one_kernel_is_shared_by_repeated_calls():
    """Two collectives of the same variant reuse one synthesized function.

    The two exchanges use disjoint signal/count windows: the public op returns
    only ``target``, so a second exchange re-reading the first's ``signal`` /
    ``recv_counts`` variable would violate the InOut use discipline (the DSL has
    no way to rebind them) — a pre-existing property of the op surface, not of
    this pass.
    """

    @pl.program
    class TwoExchanges:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            stage: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            data_a: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            signal_a: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            counts_a: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            recv_a: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            data_b: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            signal_b: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            counts_b: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            recv_b: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]:
            data_a = pld.tensor.all_to_all_v(stage, data_a, signal_a, counts_a, recv_a)
            data_b = pld.tensor.all_to_all_v(data_a, data_b, signal_b, counts_b, recv_b)
            return data_b

    result = passes.lower_l2_tensor_collectives()(TwoExchanges)

    pipeline = _get_func(result, "chip_pipeline")
    assert pipeline is not None
    assert _callee_names(pipeline) == [_KERNEL_NAME, _KERNEL_NAME]
    assert len(result.functions) == 2  # the pipeline plus one shared kernel


def test_collective_embedded_in_a_return_is_lowered():
    """``return pld.tensor.all_to_all_v(...)`` — no intermediate binding.

    The window-as-result shape makes this the shortest way to write a pipeline
    that is only the exchange, so the collective reaches the ReturnStmt without
    ever being bound to a Var.
    """

    @pl.program
    class ReturnedExchange:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            stage: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            counts: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]:
            return pld.tensor.all_to_all_v(stage, data, signal, counts, recv)

    result = passes.lower_l2_tensor_collectives()(ReturnedExchange)

    pipeline = _get_func(result, "chip_pipeline")
    assert pipeline is not None
    assert _callee_names(pipeline) == [_KERNEL_NAME]
    assert _get_func(result, _KERNEL_NAME) is not None


def test_incore_collective_is_left_alone():
    """Only orchestration bodies are this rail's business."""

    @pl.program
    class InCoreExchange:
        @pl.function(type=pl.FunctionType.InCore)
        def exchange_step(
            self,
            inp: pl.Tensor[[TOTAL, SIZE], pl.FP32],
            counts: pl.Tensor[[NR, 1], pl.INT32],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]:
            return pld.tensor.all_to_all_v(inp, data, signal, counts, recv)

    # The composite rail (LowerCompositeOps) owns this one, and it runs 25
    # passes earlier — so by this point in the pipeline an InCore collective is
    # already gone. Running this pass on one directly must not touch it, and
    # must not trip the residual check either.
    result = passes.lower_l2_tensor_collectives()(InCoreExchange)
    step = _get_func(result, "exchange_step")
    assert step is not None
    assert _callee_names(step) == [_ALL_TO_ALL_V]


def test_kernel_signature_is_canonical_not_call_site_typed():
    """`input` / `send_counts` are declared plain Tensor whatever the call passes.

    Both are Tensor-like on the public op and indistinguishable to the kernel —
    each arrives as a flat `Tensor*`. Copying the first call's types would bake
    an accidental ABI into the one dtype-keyed variant: a later call passing the
    other kind would inherit a signature whose CommCtx parameter count no longer
    matches its arguments, since MaterializeDistTensorCtx appends one per
    DistributedTensor parameter.
    """

    @pl.program
    class DistributedInput:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            stage: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            counts: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]:
            return pld.tensor.all_to_all_v(stage, data, signal, counts, recv)

    kernel = _get_func(passes.lower_l2_tensor_collectives()(DistributedInput), _KERNEL_NAME)
    assert kernel is not None
    kinds = [type(p.type).__name__ for p in kernel.params]
    # input and send_counts canonical to plain Tensor even though the call site
    # passed DistributedTensors; the other three stay distributed, which is what
    # supplies the CommCtx parameters. core_num is not a parameter at all: the
    # kernel ABI is five operands plus the ctx suffix, so `args[5]` stays the
    # CommContext the shared kernel template reads.
    assert kinds == [
        "TensorType",
        "DistributedTensorType",
        "DistributedTensorType",
        "TensorType",
        "DistributedTensorType",
    ], kinds


def test_plain_tensor_send_counts_is_rejected():
    """`send_counts` must be a DistributedTensor on this rail.

    The hand-written kernel resolves each peer's copy through `CommRemotePtr`
    for the counts pull, so only a distributed operand — the one the managed
    rail materializes as a window — is usable. The public op's deducer accepts
    a plain Tensor for the InCore composite path, where counts are consumed
    locally; accepting it here would produce garbage peer reads.
    """

    @pl.program
    class PlainCounts:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            stage: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            counts: pl.Tensor[[NR, 1], pl.INT32],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]:
            return pld.tensor.all_to_all_v(stage, data, signal, counts, recv)

    with pytest.raises(ValueError, match="send_counts must be a DistributedTensor"):
        passes.lower_l2_tensor_collectives()(PlainCounts)


def test_collective_in_a_graph_body_is_lowered():
    """A `Graph` body is orchestration-like and lands on this rail too.

    `FunctionType::Graph` derives `Role::Orchestrator` and `Level::CHIP`, so
    LowerCompositeOps already defers a collective written there. Matching only
    `FunctionType::Orchestration` here would leave it unlowered and the
    post-condition check would then reject a legal program.
    """

    @pl.program
    class GraphBody:
        @pl.function(type=pl.FunctionType.Graph)
        def chip_graph(
            self,
            stage: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            counts: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]:
            return pld.tensor.all_to_all_v(stage, data, signal, counts, recv)

    result = passes.lower_l2_tensor_collectives()(GraphBody)
    graph = _get_func(result, "chip_graph")
    assert graph is not None
    assert _callee_names(graph) == [_KERNEL_NAME]
    assert _get_func(result, _KERNEL_NAME) is not None


def test_unsupported_collective_in_a_chip_body_is_named():
    """A collective this rail cannot lower is reported here, by name.

    The orchestration-reference verifier exempts this operator family (a
    managed collective is an operator, not a function reference, so the Program
    is not expected to define it) and answers nothing beyond that, so this pass
    is the only place that answers "may this collective appear in this body". An
    unsupported one must fail here rather than ride that exemption to codegen
    and surface as an unknown operator.
    """

    @pl.program
    class ChipAllReduce:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]:
            return pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)

    with pytest.raises(ValueError, match="was not lowered"):
        passes.lower_l2_tensor_collectives()(ChipAllReduce)


def _build_exchange(nr: int, signal_stride: int, core_num):
    """One CHIP exchange over ``nr`` ranks with a ``[nr, signal_stride]`` signal."""
    total = nr * MAX_RECV

    @pl.program
    class Exchange:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            stage: pl.InOut[pld.DistributedTensor[[total, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[total, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, signal_stride], pl.INT32]],
            counts: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            recv: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[total, SIZE], pl.FP32]:
            return pld.tensor.all_to_all_v(stage, data, signal, counts, recv, core_num=core_num)

    return Exchange


def _kernel_call(program: ir.Program) -> ir.Call:
    """The single call to the synthesized builtin kernel in ``chip_pipeline``."""
    pipeline = _get_func(program, "chip_pipeline")
    assert pipeline is not None
    calls = [c for c in _collect_calls(pipeline.body) if c.op.name == _KERNEL_NAME]
    assert len(calls) == 1, [c.op.name for c in _collect_calls(pipeline.body)]
    return calls[0]


def test_single_block_request_attaches_no_launch_spec():
    """``core_num=1`` keeps the rail exactly as it was: no launch-spec attrs.

    ``B == 1`` needs no gang launch, and leaving the call untouched keeps the
    emitted orchestration byte-identical to what shipped before multi-AIV.
    """
    call = _kernel_call(passes.lower_l2_tensor_collectives()(_build_exchange(NR, 1, 1)))
    assert "core_num" not in call.attrs
    assert "sync_start" not in call.attrs


# RFC #2521 worked-example table, run through the *compiler's* fold. The HOST
# rail's entry has its own, independently-compiled copy of the formula (pinned
# in test_host_orch_distributed.py); asserting both against the same table is
# what keeps the two implementation sites from drifting. Each row is
# (P, requested L, admitted B).
_RFC_2521_BLOCK_TABLE = (
    (8, 1, 1),
    (8, 7, 7),
    (8, 8, 8),
    (8, 10, 8),
    (8, 15, 8),
    (8, 16, 16),
    (16, 7, 7),
    (16, 16, 16),
)


@pytest.mark.parametrize(("p", "requested", "blocks"), _RFC_2521_BLOCK_TABLE)
def test_launch_spec_carries_the_folded_block_count(p, requested, blocks):
    """The call asks the runtime for ``B = CalAllToAllVBlocks(P, L)``, never ``L``.

    ``(8, 10) -> 8`` is the row that distinguishes the two: the request is not a
    multiple of P, so the gang is rounded down to whole peers.
    """
    program = passes.lower_l2_tensor_collectives()(_build_exchange(p, max(blocks, 1), requested))
    call = _kernel_call(program)
    if blocks == 1:
        assert "core_num" not in call.attrs
        return
    core_num = call.attrs["core_num"]
    assert isinstance(core_num, ir.ConstInt), type(core_num).__name__
    assert core_num.value == blocks
    # The block-aware barrier needs every admitted block to start together; the
    # runtime rejects an over-capacity sync-start launch instead of part-starting it.
    assert call.attrs["sync_start"] is True


def test_launch_spec_does_not_change_the_kernel_abi():
    """Five operands + the CommCtx suffix, whatever ``core_num`` is.

    The launch spec rides the call's attrs, which are not parameters. Making
    ``core_num`` a kernel parameter once shifted the CommContext off the shared
    kernel's fixed ``args[5]`` and parked a block in ``ExchangeBarrier`` until
    the watchdog fired, so the signature must not depend on the request.
    """
    single = passes.lower_l2_tensor_collectives()(_build_exchange(4, 4, 1))
    multi = passes.lower_l2_tensor_collectives()(_build_exchange(4, 4, 4))
    expected = ["input", "target", "signal", "send_counts", "recv_counts"]
    for program in (single, multi):
        kernel = _get_func(program, _KERNEL_NAME)
        assert kernel is not None
        assert [p.name_hint for p in kernel.params] == expected
        assert len(_kernel_call(program).args) == len(expected)


def test_capacity_check_uses_the_folded_block_count_not_the_request():
    """``L=10, P=8, S=8`` is legal: ``B`` folds to 8, so a stride of 8 suffices.

    Comparing the stride against the raw request would reject this request even
    though only eight blocks launch.
    """
    call = _kernel_call(passes.lower_l2_tensor_collectives()(_build_exchange(8, 8, 10)))
    assert call.attrs["core_num"].value == 8


def test_signal_stride_below_the_launched_blocks_is_rejected():
    """``S < B`` is a defect, not something the kernel should silently clamp.

    Same condition, and the same wording, as the HOST rail's entry check.
    """
    with pytest.raises(
        ValueError, match=r"signal stride \(1\) is smaller than the admitted block count \(2\)"
    ):
        passes.lower_l2_tensor_collectives()(_build_exchange(2, 1, 2))


@pytest.mark.parametrize("requested", [2**31, 2**40])
def test_core_num_above_int32_is_rejected(requested):
    """A static request past ``INT32_MAX`` can never be honoured, so it is rejected up front.

    The launch width is a 32-bit count and the HOST entry bounds ``core_num`` to
    ``[1, INT32_MAX]`` the same way; folding it instead would overflow ``B``.
    """
    with pytest.raises(ValueError, match=r"core_num must be in \[1, 2147483647\]"):
        passes.lower_l2_tensor_collectives()(_build_exchange(NR, 1, requested))


def test_dynamic_core_num_is_rejected_with_guidance():
    """A runtime ``core_num`` is out of scope for this cut and must say so."""

    @pl.program
    class DynamicRequest:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            stage: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NR, 4], pl.INT32]],
            counts: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            n: pl.Scalar[pl.INDEX],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]:
            return pld.tensor.all_to_all_v(stage, data, signal, counts, recv, core_num=n)

    with pytest.raises(ValueError, match="a runtime core_num is not supported on this rail yet"):
        passes.lower_l2_tensor_collectives()(DynamicRequest)


def test_int8_synthesizes_int8_variant():
    """INT8 is the RFC #2521 A1 canonical payload and must share the CHIP rail."""

    @pl.program
    class Int8Exchange:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            stage: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.INT8]],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.INT8]],
            signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            counts: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], pl.INT8]:
            return pld.tensor.all_to_all_v(stage, data, signal, counts, recv)

    result = passes.lower_l2_tensor_collectives()(Int8Exchange)
    kernel = _get_func(result, "__builtin_all_to_all_v__int8")
    assert kernel is not None
    attrs = dict(kernel.attrs)
    template_vars = dict(item.split("=", 1) for item in attrs["builtin_template_vars"].split(","))
    assert template_vars == {"dtype_cpp": "int8_t"}


def _build_indexed_program(*, incore=False, cores=NR):
    @pl.program
    class Indexed:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            inp: pl.Tensor[[13, SIZE], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NR, cores], pl.INT32]],
            counts: pld.DistributedTensor[[NR, 1], pl.INT32],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            index: pl.Tensor[[NR, MAX_RECV], pl.INT32],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], pl.FP32]:
            return pld.tensor.indexed_all_to_all_v(inp, data, signal, counts, recv, index, core_num=cores)

    if incore:
        return pl.parse_program(
            ir.python_print(Indexed, format=False).replace(
                "type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator",
                "type=pl.FunctionType.InCore",
            )
        )
    return Indexed


def test_indexed_collective_l2_abi_and_dependency_directions():
    result = passes.lower_l2_tensor_collectives()(_build_indexed_program())
    name = "__builtin_indexed_all_to_all_v__fp32"
    kernel = _get_func(result, name)
    assert kernel is not None
    assert [p.name_hint for p in kernel.params] == [
        "input",
        "target",
        "signal",
        "send_counts",
        "recv_counts",
        "index",
    ]
    assert kernel.param_directions == [
        ir.ParamDirection.In,
        ir.ParamDirection.InOut,
        ir.ParamDirection.InOut,
        ir.ParamDirection.In,
        ir.ParamDirection.InOut,
        ir.ParamDirection.In,
    ]
    assert isinstance(kernel.params[0].type, ir.TensorType)
    assert isinstance(kernel.params[5].type, ir.TensorType)
    assert kernel.attrs["builtin_template_dir"] == (
        ":pypto.runtime.builtins.collectives.indexed_all_to_all_v"
    )
    main = _get_func(result, "main")
    assert main is not None
    calls = [c for c in _collect_calls(main.body) if c.op.name == name]
    assert len(calls) == 1
    arg_names = []
    for arg in calls[0].args:
        assert isinstance(arg, ir.Var)
        arg_names.append(arg.name_hint)
    assert arg_names == [
        "inp",
        "data",
        "signal",
        "counts",
        "recv",
        "index",
    ]
    # The public API must also survive printer/parser roundtrip before lowering.
    before = _build_indexed_program()
    reparsed = pl.parse_program(ir.python_print(before, format=False))
    ir.assert_structural_equal(before, reparsed)


def test_indexed_collective_rejects_incore_lowering():
    with pytest.raises(ValueError, match="InCore is unsupported"):
        passes.lower_composite_ops()(_build_indexed_program(incore=True))


def test_indexed_collective_lowers_local_receive_counts():
    source = ir.python_print(_build_indexed_program(), format=False)
    source = source.replace("recv: pl.InOut[pld.DistributedTensor", "recv: pl.InOut[pl.Tensor")
    result = passes.lower_l2_tensor_collectives()(pl.parse_program(source))
    kernel = _get_func(result, "__builtin_indexed_all_to_all_v__fp32")
    assert kernel is not None
    assert isinstance(kernel.params[4].type, ir.TensorType)
    assert kernel.param_directions[4] == ir.ParamDirection.InOut
    assert sum(isinstance(p.type, ir.DistributedTensorType) for p in kernel.params) == 2


@pytest.mark.parametrize("cores", [3, 5])
def test_indexed_collective_launches_exact_extra_local_aivs(cores):
    result = passes.lower_l2_tensor_collectives()(_build_indexed_program(cores=cores))
    main = _get_func(result, "main")
    assert main is not None
    name = "__builtin_indexed_all_to_all_v__fp32"
    calls = [c for c in _collect_calls(main.body) if c.op.name == name]
    assert len(calls) == 1
    launch_width = calls[0].attrs["core_num"]
    assert isinstance(launch_width, ir.ConstInt)
    assert launch_width.value == cores
    assert calls[0].attrs["sync_start"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
