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
        def visit_call(self, call: ir.Call) -> None:
            found.append(call)

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
    """The kernel is an AIV function carrying the five collective operands."""
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
    # supplies the CommCtx parameters.
    assert kinds == [
        "TensorType",
        "DistributedTensorType",
        "DistributedTensorType",
        "TensorType",
        "DistributedTensorType",
    ], kinds


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

    The orchestration-reference verifier exempts the whole ``pld.`` operator
    namespace (a `pld.*` callee is not a function reference), so this pass is
    the only place that answers "may this collective appear in this body". An
    unsupported one must fail here rather than reach codegen as an unknown
    operator.
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


def test_multi_core_request_is_rejected():
    """core_num > 1 is not implemented yet and must fail loudly, not silently."""
    with pytest.raises(ValueError, match="only core_num=1"):
        passes.lower_l2_tensor_collectives()(_build_program(core_num=2))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
