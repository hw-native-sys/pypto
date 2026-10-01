# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the ergonomic ``pld.*`` collective wrappers (``collective_api.py``).

Covers auto-signal allocation (shape per op, fresh-per-call unique names),
mesh-vs-ring signal handling, kwarg passthrough, and parser resolution of the
``pld.<op>`` short forms inside a host-orchestration program body.
"""

from typing import Any, cast

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import ir
from pypto.pypto_core import ir as _ir
from pypto.pypto_core.ir import Call, ConstInt, ShapedType

SIZE = 8


def _window(shape, dtype, name) -> pld.DistributedTensor:
    """Build a window-bound DistributedTensor in pure Python (explicit alloc name)."""
    buf = pld.tensor.alloc_window_buffer(shape, dtype=dtype, name=name)
    return pld.tensor.window(buf, shape, dtype=dtype)


def _as_call(result) -> Call:
    """Return the underlying IR Call of a wrapper result (narrowed from Expr)."""
    return cast(Call, result._expr)


def _shape_of(arg) -> list[int]:
    """Return the per-rank shape of a DistributedTensor-typed argument as ints."""
    dist_type = cast(ShapedType, cast(Call, arg).type)
    return [int(cast(ConstInt, d).value) for d in dist_type.shape]


def _signal_alloc_name(call: Call, signal_arg_index: int) -> str:
    """Extract the alloc_window_buffer name backing a signal argument."""
    window_call = cast(Call, call.args[signal_arg_index])
    alloc_call = cast(Call, window_call.args[0])  # window(buf, ...) -> buf is the alloc call
    return cast(str, alloc_call.kwargs["name"])


class TestAllReduceSignal:
    def test_mesh_emits_no_explicit_signal(self):
        target = _window([1, SIZE], pl.FP32, "t")
        call = _as_call(pld.all_reduce(target, op=pld.ReduceOp.Sum))
        assert call.op.name == _ir.get_op("pld.tensor.allreduce").name
        # Host synthesis path: no explicit signal argument.
        assert len(call.args) == 1
        assert call.kwargs["op"] == int(pld.ReduceOp.Sum)

    def test_ring_allocates_static_signal_shape(self):
        target = _window([1, SIZE], pl.FP32, "t")
        call = _as_call(pld.all_reduce(target, mode="ring", nranks=2))
        assert call.op.name == _ir.get_op("pld.tensor.allreduce").name
        # Ring signal is [2*(NR-1)+1, NR] = [3, 2] for NR=2.
        assert len(call.args) == 2
        assert _shape_of(call.args[1]) == [3, 2]
        assert cast(ShapedType, call.args[1].type).dtype == pl.INT32
        assert call.kwargs["mode"] == "ring"

    def test_ring_requires_static_nranks(self):
        target = _window([1, SIZE], pl.FP32, "t")
        # Widened to Any: the Literal overloads make mode="ring" without nranks a
        # static type error — this exercises the runtime guard for DSL code that
        # is not type-checked.
        with pytest.raises(ValueError, match="nranks"):
            cast(Any, pld.all_reduce)(target, mode="ring")

    def test_mesh_rejects_nranks(self):
        target = _window([1, SIZE], pl.FP32, "t")
        with pytest.raises(ValueError, match="nranks"):
            cast(Any, pld.all_reduce)(target, nranks=2)

    def test_invalid_mode_rejected(self):
        target = _window([1, SIZE], pl.FP32, "t")
        with pytest.raises(ValueError, match="mesh"):
            cast(Any, pld.all_reduce)(target, mode="tree")

    def test_ring_rejects_non_sum(self):
        target = _window([1, SIZE], pl.FP32, "t")
        with pytest.raises(ValueError, match="Sum"):
            pld.all_reduce(target, mode="ring", nranks=2, op=pld.ReduceOp.Max)

    def test_ring_rejects_non_fp32(self):
        target = _window([1, SIZE], pl.FP16, "t")
        with pytest.raises(ValueError, match="FP32"):
            pld.all_reduce(target, mode="ring", nranks=2)

    def test_ring_rejects_non_positive_nranks(self):
        target = _window([1, SIZE], pl.FP32, "t")
        with pytest.raises(ValueError, match="positive"):
            cast(Any, pld.all_reduce)(target, mode="ring", nranks=0)


class TestPassthrough:
    def test_broadcast_root_passthrough(self):
        target = _window([1, SIZE], pl.FP32, "t")
        call = _as_call(pld.broadcast(target, root=0))
        assert call.op.name == _ir.get_op("pld.tensor.broadcast").name
        assert call.kwargs["root"] == 0
        # target + auto signal; broadcast builtin requires a rank-1 signal.
        assert len(call.args) == 2
        assert len(cast(ShapedType, call.args[1].type).shape) == 1

    def test_reduce_scatter_op_passthrough(self):
        target = _window([2, SIZE], pl.FP32, "t")
        call = _as_call(pld.reduce_scatter(target, op=pld.ReduceOp.Sum))
        assert call.op.name == _ir.get_op("pld.tensor.reduce_scatter").name
        assert call.kwargs["op"] == int(pld.ReduceOp.Sum)
        # target + auto signal; reduce_scatter builtin requires a rank-1 signal.
        assert len(call.args) == 2
        assert len(cast(ShapedType, call.args[1].type).shape) == 1

    def test_reduce_scatter_rejects_non_sum(self):
        target = _window([2, SIZE], pl.FP32, "t")
        with pytest.raises(ValueError, match="Sum"):
            pld.reduce_scatter(target, op=pld.ReduceOp.Max)

    def test_all_to_all_and_all_gather_delegate(self):
        inp = _window([2, SIZE], pl.FP32, "inp")
        target = _window([2, SIZE], pl.FP32, "tgt")
        a2a = _as_call(pld.all_to_all(inp, target))
        assert a2a.op.name == _ir.get_op("pld.tensor.all_to_all").name
        assert len(a2a.args) == 3  # input + target + signal

        local = _window([1, SIZE], pl.FP32, "loc")
        gather = _as_call(pld.all_gather(local, target))
        assert gather.op.name == _ir.get_op("pld.tensor.allgather").name
        assert len(gather.args) == 3

    def test_all_to_all_v_delegates_with_static_nranks(self):
        inp = _window([2 * SIZE, SIZE], pl.FP32, "inp")
        target = _window([2 * SIZE, SIZE], pl.FP32, "tgt")
        send = _window([2], pl.INT32, "send")
        recv = _window([2, 1], pl.INT32, "recv")
        call = _as_call(pld.all_to_all_v(inp, target, send, recv, nranks=2))
        assert call.op.name == _ir.get_op("pld.tensor.all_to_all_v").name
        # arg order: input, target, signal, send_counts, recv_counts.
        assert len(call.args) == 5
        # signal is [nranks, 1] = [2, 1] INT32.
        assert _shape_of(call.args[2]) == [2, 1]
        assert cast(ShapedType, call.args[2].type).dtype == pl.INT32

    def test_all_to_all_v_rejects_non_positive_nranks(self):
        inp = _window([2 * SIZE, SIZE], pl.FP32, "inp")
        target = _window([2 * SIZE, SIZE], pl.FP32, "tgt")
        send = _window([2], pl.INT32, "send")
        recv = _window([2, 1], pl.INT32, "recv")
        with pytest.raises(ValueError, match="positive"):
            pld.all_to_all_v(inp, target, send, recv, nranks=0)

    def test_barrier_requires_covered_signal(self):
        signal = _window([2], pl.INT32, "sig")
        call = _as_call(pld.barrier(signal))
        assert call.op.name == _ir.get_op("pld.tensor.barrier").name
        assert len(call.args) == 1  # the user-provided signal only


class TestFreshSignal:
    def test_signal_allocated_fresh_per_call(self):
        names = []
        for i in range(2):
            local = _window([1, SIZE], pl.FP32, f"loc{i}")
            target = _window([2, SIZE], pl.FP32, f"tgt{i}")
            call = _as_call(pld.all_gather(local, target))
            names.append(_signal_alloc_name(call, signal_arg_index=2))
        assert len(set(names)) == 2, "signals must be fresh (unique) per call"
        assert all(n.startswith("__auto_") for n in names)

    def test_signal_shape_is_rank2_world_size(self):
        local = _window([1, SIZE], pl.FP32, "loc")
        target = _window([2, SIZE], pl.FP32, "tgt")
        call = _as_call(pld.all_gather(local, target))
        # signal = [world_size(), 1] — dynamic NR, static second extent 1.
        sig_type = cast(ShapedType, call.args[2].type)
        assert int(cast(ConstInt, sig_type.shape[1]).value) == 1


@pl.program
class _HostAllReduce:
    @pl.function(type=pl.FunctionType.InCore)
    def publish_step(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
        return pl.store(pl.load(inp, [0, 0], [1, SIZE]), [0, 0], data)

    @pl.function(type=pl.FunctionType.Orchestration)
    def publish_orch(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
        return self.publish_step(inp, data)

    @pl.function(type=pl.FunctionType.InCore)
    def consume_step(
        self,
        data: pld.DistributedTensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        return pl.store(pl.load(data, [0, 0], [1, SIZE]), [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def consume_orch(
        self,
        data: pld.DistributedTensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        return self.consume_step(data, out)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[2, 1, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[2, 1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[2, 1, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
            self.publish_orch(inputs[r], data, device=r)
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        data = pld.all_reduce(data, op=pld.ReduceOp.Sum)
        for r in pl.range(pld.world_size()):
            self.consume_orch(data, outputs[r], device=r)
        return outputs


@pl.program
class _HostAllGather:
    """Program exercising a signal-bearing wrapper (pld.all_gather)."""

    @pl.function(type=pl.FunctionType.InCore)
    def publish_step(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        local: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
        return pl.store(pl.load(inp, [0, 0], [1, SIZE]), [0, 0], local)

    @pl.function(type=pl.FunctionType.Orchestration)
    def publish_orch(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        local: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
        return self.publish_step(inp, local)

    @pl.function(type=pl.FunctionType.InCore)
    def consume_step(
        self,
        target: pld.DistributedTensor[[2, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        return pl.store(pl.load(target, [0, 0], [1, SIZE]), [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def consume_orch(
        self,
        target: pld.DistributedTensor[[2, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        return self.consume_step(target, out)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[2, 1, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[2, 1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[2, 1, SIZE], pl.FP32]:
        local_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
        target_buf = pld.alloc_window_buffer(2 * SIZE * pl.FP32.get_byte())
        for r in pl.range(2):
            local = pld.window(local_buf, [1, SIZE], dtype=pl.FP32)
            self.publish_orch(inputs[r], local, device=r)
        local = pld.window(local_buf, [1, SIZE], dtype=pl.FP32)
        target = pld.window(target_buf, [2, SIZE], dtype=pl.FP32)
        target = pld.all_gather(local, target)  # auto signal: __auto_allgather_<n>
        for r in pl.range(2):
            self.consume_orch(target, outputs[r], device=r)
        return outputs


class TestParserResolution:
    def test_pld_all_reduce_resolves_in_host_body(self):
        """The parser resolves the pld.all_reduce short form and round-trips."""
        # as_python() parses the program body; an unresolved pld.all_reduce
        # would raise "Unknown distributed operation" here. The printer
        # expands the wrapper to the canonical pld.tensor.allreduce IR op.
        printed = _HostAllReduce.as_python()
        assert "pld.tensor.allreduce(" in printed
        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(_HostAllReduce, reparsed)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "as_python() prints the auto signal as an inline "
            "window(alloc_window_buffer(...)) with the name dropped; parse_program "
            "requires alloc_window_buffer as the RHS of a simple assignment, so "
            "signal-bearing wrappers cannot round-trip yet (python_printer hoist fix)."
        ),
    )
    def test_signal_bearing_wrapper_round_trips(self):
        """A wrapper with an auto-allocated signal must survive print→reparse."""
        printed = _HostAllGather.as_python()
        assert "pld.tensor.allgather(" in printed
        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(_HostAllGather, reparsed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
