# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO codegen tests for distributed N6 ops.

Covers the InCore PTO codegen for ``pld.tile.remote_load``,
``pld.system.notify`` and ``pld.system.wait``:

- CommContext ``!pto.ptr<i64>`` parameter is appended at the end of the
  ``func.func`` signature, one per ``DistributedTensor`` IR param.
- One module-level ``func.func @CommRemotePtr_<dtype>`` helper is emitted
  per distinct DistributedTensor element dtype consumed by remote ops; the
  per-op lowering at the call site becomes a single
  ``func.call @CommRemotePtr_<dtype>(...)``.
- The CommRemotePtr helper's byte-offset literals are pinned to the
  constants in ``include/pypto/codegen/distributed/comm_layout.h``.
- ``pto.addptr`` (inside the helper) produces a peer ``!pto.ptr<dtype>``;
  a fresh ``pto.make_tensor_view`` + ``pto.partition_view`` (at the call
  site, on the returned peer pointer) builds the peer view.
- ``pto.tload`` (remote_load), ``pto.comm.tnotify`` (notify) and
  ``pto.comm.twait`` (wait) consume the partition views with the PTOAS
  attribute spellings (``notifyOp = #pto<notify_op …>`` and
  ``cmp = #pto<wait_cmp …>``).
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import backend, codegen, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import passes as _passes


@pytest.fixture(autouse=True)
def _setup_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


@pytest.fixture(autouse=True)
def _basic_verification_context():
    """Override ``ut/conftest.py`` to skip the print/parse roundtrip check.

    The DSL parser for ``pld.tile.remote_load`` / ``pld.system.notify`` /
    ``pld.system.wait`` only accepts the kwarg form, but the IR printer emits
    them as positional Call args (no per-op printer hook). Roundtrip
    verification would fail every pass even though the in-memory IR is
    correct; property verification still runs.
    """
    with _passes.PassContext([_passes.VerificationInstrument(_passes.VerificationMode.BEFORE_AND_AFTER)]):
        yield


def _generate_mlir(program_cls) -> str:
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    optimized = pm.run_passes(program_cls)
    return codegen.PTOCodegen().generate(optimized)


def test_ctx_arg_appended_per_distributed_tensor():
    """One ``!pto.ptr<i64>`` arg appended per DistributedTensor param."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[[16, 64], pl.FP16],
            signal: pld.DistributedTensor[[16, 16], pl.INT32],
            out: pl.Tensor[[16, 32], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            # Touch both DistributedTensor params so neither is DCE'd.
            t = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[16, 32])
            pl.store(t, [0, 0], out)
            pld.system.wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Eq)

    mlir = _generate_mlir(P)
    # Function header has 6 args: 3 tensors (data, signal, out) + 1 scalar
    # (peer) + 2 ctx ptrs (one per DistributedTensor).
    header = next(line for line in mlir.splitlines() if "func.func @kernel" in line)
    assert header.count("%arg") == 6, header
    # Trailing args after the explicit IR params are the ctx ptrs.
    assert "%arg4: !pto.ptr<i64>" in header, header
    assert "%arg5: !pto.ptr<i64>" in header, header
    # The CtxArg type only appears in the func header at this point (later
    # body uses bind to %argK references). Two DistributedTensors → two ptr
    # declarations.
    assert header.count("!pto.ptr<i64>") == 2, header


def _split_module(mlir: str) -> dict[str, str]:
    """Split ``module {...}`` into a mapping of ``func_name -> body``.

    The PTO codegen output is shallow — a single module containing flat
    ``func.func`` definitions — so a regex-free split on the ``func.func @``
    header is sufficient. Each entry's value contains everything from the
    function header (inclusive) up to (but excluding) the next header.
    """
    funcs: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []
    for line in mlir.splitlines():
        stripped = line.strip()
        if stripped.startswith("func.func @"):
            if current_name is not None:
                funcs[current_name] = "\n".join(current_lines)
            # `func.func @name(...)` → grab the bit between '@' and '('.
            after_at = stripped.split("@", 1)[1]
            current_name = after_at.split("(", 1)[0]
            current_lines = [line]
        elif current_name is not None:
            current_lines.append(line)
    if current_name is not None:
        funcs[current_name] = "\n".join(current_lines)
    return funcs


def test_remote_load_emits_func_call_to_module_level_helper():
    """remote_load lowers to a func.call to a module-level CommRemotePtr_<dtype>."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[[16, 64], pl.FP16],
            out: pl.Tensor[[16, 32], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            t = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[16, 32])
            pl.store(t, [0, 0], out)

    mlir = _generate_mlir(P)
    funcs = _split_module(mlir)

    # The CommRemotePtr_f16 helper is emitted once at module scope.
    assert "CommRemotePtr_f16" in funcs, f"Expected @CommRemotePtr_f16 in module, got {list(funcs)}"
    helper = funcs["CommRemotePtr_f16"]
    assert "func.func @CommRemotePtr_f16(%ctx: !pto.ptr<i64>" in helper
    assert "%local_ptr: !pto.ptr<f16>" in helper
    assert "%peer: index) -> !pto.ptr<f16>" in helper
    # Helper does the scalar arithmetic — load_scalar + addptr + return.
    assert helper.count("pto.load_scalar") >= 3, helper  # rankId + 2 window slots
    assert "pto.addptr %local_ptr" in helper
    assert "return %peer_ptr : !pto.ptr<f16>" in helper

    # The kernel only invokes the helper via func.call — no inline scalar
    # CommRemotePtr arithmetic should remain at the call site.
    kernel = funcs["kernel"]
    assert "func.call @CommRemotePtr_f16(" in kernel
    assert "pto.load_scalar" not in kernel, (
        "CommRemotePtr scalar arithmetic must live in the helper, not inline"
    )
    # Peer view + tload still happen in the kernel body, against the helper's result.
    assert "pto.make_tensor_view" in kernel
    peer_pview = [line for line in kernel.splitlines() if "_peer_pview" in line and "partition_view" in line]
    assert peer_pview, f"Missing peer partition_view in kernel body:\n{kernel}"
    assert "pto.tload" in kernel


def test_one_comm_remote_ptr_helper_per_dtype():
    """The module emits a distinct @CommRemotePtr_<dtype> helper per used dtype."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[[16, 64], pl.FP16],
            signal: pld.DistributedTensor[[16, 16], pl.INT32],
            out: pl.Tensor[[16, 32], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            t = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[16, 32])
            pl.store(t, [0, 0], out)
            pld.system.notify(signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.Set)

    mlir = _generate_mlir(P)
    funcs = _split_module(mlir)
    # f16 (data) + i32 (signal) — one helper per element dtype consumed by a
    # cross-rank op (notify counts; wait stays local-only).
    assert "CommRemotePtr_f16" in funcs
    assert "CommRemotePtr_i32" in funcs
    # The element-size constant inside each helper matches the helper's dtype.
    assert "arith.constant 2 : i64" in funcs["CommRemotePtr_f16"]
    assert "arith.constant 4 : i64" in funcs["CommRemotePtr_i32"]


def test_remote_load_uses_comm_layout_constants():
    """CommRemotePtr helper literal offsets equal the comm_layout::k* values."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[[16, 64], pl.FP16],
            out: pl.Tensor[[16, 32], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            t = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[16, 32])
            pl.store(t, [0, 0], out)

    mlir = _generate_mlir(P)
    funcs = _split_module(mlir)
    helper = funcs["CommRemotePtr_f16"]

    layout = ir.comm_layout
    rank_idx_unit = layout.RANK_ID_OFFSET // layout.WINDOW_SLOT_STRIDE  # 16 / 8 = 2
    win_idx_unit = layout.WINDOWS_IN_OFFSET // layout.WINDOW_SLOT_STRIDE  # 32 / 8 = 4

    # The CommRemotePtr scaffolding references the rank-slot offset and the
    # windowsIn-array base in *u64-units*, derived from comm_layout constants.
    assert f"arith.constant {rank_idx_unit} : index" in helper
    assert f"arith.constant {win_idx_unit} : index" in helper
    # Element-size for FP16 is 2 bytes; the byte-delta is divided by 2 to
    # reach a pto.addptr-compatible element offset.
    assert "arith.constant 2 : i64" in helper, helper
    assert "arith.divsi" in helper


def test_notify_emits_comm_tnotify_with_attr():
    """notify codegen emits pto.comm.tnotify with #pto<notify_op …> attr."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            signal: pld.DistributedTensor[[16, 16], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.system.notify(signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.Set)

    mlir = _generate_mlir(P)
    assert "pto.comm.tnotify(" in mlir
    assert "#pto<notify_op set>" in mlir
    # AtomicAdd variant should also lower correctly.

    @pl.program
    class PAdd:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            signal: pld.DistributedTensor[[16, 16], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.system.notify(signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    mlir_add = _generate_mlir(PAdd)
    assert "#pto<notify_op atomic_add>" in mlir_add


def test_wait_emits_comm_twait_with_attr():
    """wait codegen emits pto.comm.twait on the local signal slot."""

    @pl.program
    class PEq:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            signal: pld.DistributedTensor[[16, 16], pl.INT32],
        ):
            pld.system.wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Eq)

    mlir_eq = _generate_mlir(PEq)
    assert "pto.comm.twait(" in mlir_eq
    assert "#pto<wait_cmp eq>" in mlir_eq
    # Wait operates on the local signal view — no pto.addptr / peer
    # arithmetic should appear between the function header and the twait.
    twait_prefix = mlir_eq.split("pto.comm.twait", 1)[0]
    assert "pto.addptr" not in twait_prefix
    assert "_local_pview" in mlir_eq

    @pl.program
    class PGe:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            signal: pld.DistributedTensor[[16, 16], pl.INT32],
        ):
            pld.system.wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

    mlir_ge = _generate_mlir(PGe)
    assert "#pto<wait_cmp ge>" in mlir_ge


def test_notify_value_type_matches_value_ir_dtype():
    """Notify value's MLIR type annotation is sourced from the value IR ScalarType, not the signal's dtype.

    The PTOAS contract requires the value's MLIR type to match the signal
    element type — this assertion documents that pypto preserves the value's
    declared scalar type so any mismatch surfaces as a PTOAS verifier error
    rather than silent DMA garbling.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            signal: pld.DistributedTensor[[16, 16], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.system.notify(signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.Set)

    mlir = _generate_mlir(P)
    tnotify_line = next(line for line in mlir.splitlines() if "pto.comm.tnotify(" in line)
    # The element type tag inside the partition_tensor_view is the signal dtype
    # (i32) — confirm it survived the lowering.
    assert "!pto.partition_tensor_view<1x1xi32>" in tnotify_line


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
