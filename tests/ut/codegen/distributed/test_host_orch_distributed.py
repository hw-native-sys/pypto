# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Text tests for distributed host_orch python codegen.

Covers four orthogonal pieces of the host_orch emit:

1. Programs with at least one CommGroup wrap the body in
   ``with orch.allocate_domain(name=..., workers=..., window_size=...,
   buffers=[CommBufferSpec(...)]) as __comm_d0:``.
2. DistributedTensor formal → ``add_tensor(ContinuousTensor.make(data=__comm_d0[<r>]
   .buffer_ptrs["<name>"], shapes=..., dtype=..., child_memory=True), ...)``.
3. Per-DistributedTensor trailing ctx scalar:
   ``add_scalar(__comm_d0[<r>].device_ctx)`` placed AFTER all tensor adds,
   in IR-arg order (matches the incore func.func trailing-ctx segment).
4. dispatch ``device=`` attr → ``submit_next_level(..., worker=<r>)`` kwarg.

Plus regressions:

* ``pld.system.world_size()`` lowers to the ``world_size`` kwarg in any expr
  context (e.g. ``pl.range(...)``).
* Comm-less L3 dispatch (no ``device=``) still emits ``submit_next_level(...,
  config)`` without the ``worker=`` kwarg AND without an ``allocate_domain``
  wrapper, preserving binary compatibility with existing L3 demos.
"""

import re

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import codegen
from pypto.pypto_core import passes  # match the import path used by ut/conftest.py

SIZE = 64


@pytest.fixture(autouse=True)
def pass_verification_context():
    """Override ``ut/conftest.py``'s autouse roundtrip-verification fixture.

    ``CollectCommGroups`` materialises ``DistributedTensorType.window_buffer_``
    back-references that the printer / parser pair has no surface syntax for —
    the roundtrip check would fail despite the in-memory IR being correct.
    Mirrors the same override in
    [tests/ut/ir/transforms/test_collect_comm_groups.py](../../ir/transforms/test_collect_comm_groups.py).
    The fixture name MUST be ``pass_verification_context`` to shadow the
    conftest's same-named autouse fixture (pytest fixture override semantics).
    """
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        yield


def _lower(program) -> str:
    """Apply ``CollectCommGroups`` (so ``DistributedTensorType.window_buffer_``
    is populated), then run distributed codegen directly."""
    program = passes.collect_comm_groups()(program)
    cg = codegen.DistributedCodegen()
    return cg.generate(program)


# ---------------------------------------------------------------------------
# Positive: DistributedTensor formals + device= → with orch.allocate_domain
# + ContinuousTensor.make + add_scalar(ctx) + worker= + world_size lowering
# ---------------------------------------------------------------------------


def test_dist_tensor_formal_emits_continuous_tensor_make():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[SIZE], pl.FP32]],
            data: pld.DistributedTensor[[SIZE], pl.FP32],
        ) -> pl.Tensor[[SIZE], pl.FP32]:
            return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[2, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[2, SIZE], pl.FP32]],
        ) -> pl.Tensor[[2, SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(SIZE * 4)
            data = pld.window(data_buf, [SIZE], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                # Manually hoist the per-rank slices so the tensor.slice
                # host_orch handler lands in AssignStmt position (in production
                # this hoisting is done by FlattenCallExpr).
                inp_r = inputs[r]
                out_r = outputs[r]
                self.chip_orch(inp_r, out_r, data, device=r)
            return outputs

    code = _lower(Prog)

    # for r in range(..., world_size, ...):  ← world_size lowering in loop bound.
    assert re.search(r"for \w+ in range\(.*\bworld_size\b.*\):", code), code
    # ContinuousTensor.make for the DistributedTensor formal — keyed on
    # the alloc op's LHS name_hint (``data_buf``).
    assert re.search(
        r'ContinuousTensor\.make\(data=__comm_d0\[\w+\]\.buffer_ptrs\["data_buf"\],'
        r" shapes=\(64,\), dtype=DataType\.FLOAT32, child_memory=True\)",
        code,
    ), code
    # Trailing per-DistributedTensor ctx scalar — same rank index as
    # the ContinuousTensor.make above.
    assert re.search(r"\.add_scalar\(__comm_d0\[\w+\]\.device_ctx\)", code), code
    # ``device=r`` → ``worker=r`` kwarg on submit_next_level.
    assert re.search(r"submit_next_level\(callables\[\"chip_orch\"\],.*config, worker=\w+\)", code), code


def test_two_dist_tensor_formals_emit_two_ctx_scalars():
    """Two ``DistributedTensor`` formals emit two
    ``add_scalar(__comm_d0[r].device_ctx)`` in IR-arg order, both after all
    ``add_tensor`` lines (PTOParam tensor-first invariant + trailing-ctx
    convention)."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            data: pld.DistributedTensor[[SIZE], pl.FP32],
            signal: pld.DistributedTensor[[1], pl.INT32],
        ) -> pl.Tensor[[1], pl.INT32]:
            return signal  # type: ignore[return-value]

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self) -> pl.Tensor[[1], pl.INT32]:
            data_buf = pld.alloc_window_buffer(SIZE * 4)
            signal_buf = pld.alloc_window_buffer(4)
            data = pld.window(data_buf, [SIZE], dtype=pl.FP32)
            signal = pld.window(signal_buf, [1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, signal, device=r)
            return signal  # type: ignore[return-value]

    code = _lower(Prog)

    # Two ContinuousTensor.make lines — one per DistributedTensor formal.
    cont_makes = re.findall(
        r'ContinuousTensor\.make\(data=__comm_d0\[\w+\]\.buffer_ptrs\["([^"]+)"\],'
        r" shapes=(\([^)]*\)), dtype=DataType\.([A-Z0-9]+),",
        code,
    )
    assert len(cont_makes) == 2, code
    # IR-arg order: data first (FP32, name ``data_buf``), then signal (INT32, name ``signal_buf``).
    assert cont_makes[0] == ("data_buf", "(64,)", "FLOAT32"), cont_makes
    assert cont_makes[1] == ("signal_buf", "(1,)", "INT32"), cont_makes
    # Two add_scalar lines — same rank index, same order as the two formals.
    scalars = re.findall(r"\.add_scalar\(__comm_d0\[(\w+)\]\.device_ctx\)", code)
    assert len(scalars) == 2, code
    assert scalars[0] == scalars[1], scalars  # both subscript the same rank


def test_const_device_kwarg_renders_literal_worker():
    """``device=0`` (ConstInt) lowers to ``worker=0`` literal — both the
    ``submit_next_level`` worker kwarg AND the ``__comm_d0[0]`` subscript.

    The dispatch call is in an ``AssignStmt`` (rather than the outer ``return``)
    because DistributedCodegen routes chip-orch dispatches through
    ``TryEmitHierarchyCall`` only from the AssignStmt visitor — see
    [distributed_codegen.cpp:248-280](../../../src/codegen/distributed/distributed_codegen.cpp#L248).
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            out: pl.Out[pl.Tensor[[SIZE], pl.FP32]],
            data: pld.DistributedTensor[[SIZE], pl.FP32],
        ) -> pl.Tensor[[SIZE], pl.FP32]:
            return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, out: pl.Out[pl.Tensor[[SIZE], pl.FP32]]) -> pl.Tensor[[SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(SIZE * 4)
            data = pld.window(data_buf, [SIZE], dtype=pl.FP32)
            result: pl.Tensor[[SIZE], pl.FP32] = self.chip_orch(out, data, device=0)
            return result

    code = _lower(Prog)
    assert "submit_next_level" in code and "worker=0" in code, code
    assert "__comm_d0[0].buffer_ptrs" in code, code
    assert "__comm_d0[0].device_ctx" in code, code


def test_comm_group_program_emits_allocate_domain_with_block():
    """Programs with ``pld.alloc_window_buffer`` emit a
    ``with orch.allocate_domain(...)`` wrapping the for-loop body, with the
    same ``__comm_d0`` handle used for all ``buffer_ptrs`` / ``device_ctx``
    accesses below."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[SIZE], pl.FP32]) -> pl.Tensor[[SIZE], pl.FP32]:
            return data  # type: ignore[return-value]

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self) -> pl.Tensor[[SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(SIZE * 4)
            data = pld.window(data_buf, [SIZE], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            return data  # type: ignore[return-value]

    code = _lower(Prog)
    # The with-block opens with the literal spec list and binds the handle.
    assert re.search(r"with orch\.allocate_domain\(", code), code
    assert re.search(r'name="comm_d0",', code), code
    # Empty CommGroup.devices_ (this program declares no explicit subset on
    # the alloc) lowers to `workers=[*range(world_size)]` — resolved at
    # orch_fn time against the runner-bound `world_size` kwarg.
    assert re.search(r"workers=\[\*range\(world_size\)\],", code), code
    # window_size is the sum of all slot nbytes expressions, each parenthesised.
    # Single slot of 256 bytes → `window_size=(256),`.
    assert re.search(r"window_size=\(256\),", code), code  # 64 * 4
    assert re.search(
        r'CommBufferSpec\(name="data_buf", dtype="opaque", count=256, nbytes=256\),',
        code,
    ), code
    assert "as __comm_d0:" in code, code
    # All buffer / device_ctx lookups go through the handle; the legacy
    # ``contexts`` parameter must not appear anywhere.
    assert "contexts[" not in code, code
    assert re.search(r"__comm_d0\[\w+\]\.buffer_ptrs", code), code


# ---------------------------------------------------------------------------
# Negative / regression
# ---------------------------------------------------------------------------


def test_comm_less_dispatch_omits_worker_kwarg():
    """Comm-less L3 dispatch (no ``device=`` attr) still emits ``submit_next_level(...,
    config)`` without trailing ``worker=`` and without an ``allocate_domain``
    wrapper — byte-compatible with existing L3 demos (test_l3_distributed.py /
    test_l3_parallel_reduce.py)."""

    @pl.program
    class Prog:
        @pl.function(level=pl.Level.CHIP, role=pl.Role.SubWorker)
        def chip_worker(self, x: pl.Tensor[[SIZE], pl.FP32]) -> pl.Tensor[[SIZE], pl.FP32]:
            y: pl.Tensor[[SIZE], pl.FP32] = pl.add(x, x)
            return y

        @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip_orch(self, x: pl.Tensor[[SIZE], pl.FP32]) -> pl.Tensor[[SIZE], pl.FP32]:
            y: pl.Tensor[[SIZE], pl.FP32] = self.chip_worker(x)
            return y

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, x: pl.Tensor[[SIZE], pl.FP32]) -> pl.Tensor[[SIZE], pl.FP32]:
            y: pl.Tensor[[SIZE], pl.FP32] = self.chip_orch(x)
            return y

    code = _lower(Prog)
    # The dispatch shape stays intact; the comm-less path emits no wrapper
    # and no ctx-scalar / ContinuousTensor.make / handle subscript.
    assert "submit_next_level(" in code, code
    assert "worker=" not in code, code
    assert "ContinuousTensor.make" not in code, code
    assert "__comm_d0[" not in code, code
    assert "allocate_domain" not in code, code
    assert "with orch" not in code, code


def test_world_size_lowers_to_kwarg_in_expression_context():
    """``pld.system.world_size()`` is recognised by the expression visitor
    regardless of where it appears (loop bound, allocation arg, etc.).

    Both the alloc-size form (``alloc_window_buffer(pld.world_size() * 4)`` —
    flows through ``EmitCommDomainAllocations``' per-slot ``VisitExpr``) and
    the loop-bound form (``pl.range(pld.world_size())``) must lower to a bare
    reference to the ``world_size`` kwarg in the emitted python.
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, signal: pld.DistributedTensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
            return signal  # type: ignore[return-value]

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self) -> pl.Tensor[[1], pl.INT32]:
            # alloc size threading pld.world_size() — exercises the alloc-arg path.
            buf = pld.alloc_window_buffer(pld.world_size() * 4)
            signal = pld.window(buf, [1], dtype=pl.INT32)
            # loop bound — exercises the for-stop path.
            for r in pl.range(pld.world_size()):
                self.chip_orch(signal, device=r)
            return signal  # type: ignore[return-value]

    code = _lower(Prog)
    # The literal ``pld.system.world_size()`` call must NOT survive into the
    # emitted python.
    assert "pld.system.world_size" not in code, code
    # Loop bound: bare `world_size` reference.
    assert re.search(r"for \w+ in range\(.*\bworld_size\b.*\):", code), code
    # Alloc-size: the CommBufferSpec / window_size args carry a dynamic
    # `world_size * 4` expression in place of a constant nbytes literal.
    assert re.search(r"window_size=\(.*\bworld_size\b.*\),", code), code
    assert re.search(r"CommBufferSpec\(.*\bworld_size\b.*\),", code), code
    # The legacy ``len(contexts)`` form must not survive.
    assert "len(contexts)" not in code, code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
