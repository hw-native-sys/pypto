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
- One module-level ``func.func @CommRemoteOffset_<dtype>`` helper is
  emitted per distinct element dtype consumed by remote ops. The helper
  reads the CommContext field, computes the byte→element delta between
  the local rank's window slice and the peer's slice, and returns it as
  an ``index``. Each remote-op call site is a single
  ``func.call @CommRemoteOffset_<dtype>(ctx, peer) -> index`` followed by
  ``pto.addptr`` + ``pto.make_tensor_view`` in the user kernel.
- ``pto.addptr`` and ``pto.make_tensor_view`` MUST live at the call site,
  not in the helper: PTOAS verifies per-function that ``addptr`` directly
  feeds ``make_tensor_view`` / ``initialize_l2g2l_pipe(gm_addr)`` /
  ``load|store_scalar``, AND ``make_tensor_view`` lowers to a strided
  memref whose layout cannot be encoded in a ``!pto.tensor_view<…>``
  return type — so the view cannot be returned across a func boundary
  either. Returning the offset is the only shape that satisfies both
  constraints while still sharing the CommContext reads.
- The helper's byte-offset literals are pinned to the constants in
  ``include/pypto/codegen/distributed/comm_layout.h``.
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


@pytest.fixture(autouse=True)
def _setup_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


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

    Handles both ``func.func @name(...)`` and ``func.func private @name(...)``.
    """
    funcs: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []
    for line in mlir.splitlines():
        stripped = line.strip()
        if stripped.startswith("func.func ") and "@" in stripped:
            if current_name is not None:
                funcs[current_name] = "\n".join(current_lines)
            after_at = stripped.split("@", 1)[1]
            current_name = after_at.split("(", 1)[0]
            current_lines = [line]
        elif current_name is not None:
            current_lines.append(line)
    if current_name is not None:
        funcs[current_name] = "\n".join(current_lines)
    return funcs


def test_remote_load_emits_func_call_to_offset_helper_with_addptr_at_call_site():
    """remote_load lowers to func.call @CommRemoteOffset_<dtype> + addptr + make_tensor_view at call site."""

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

    # Helper signature: (ctx, peer) → index. No local_ptr arg, no addptr,
    # no make_tensor_view inside — those live at the call site.
    helper_name = "CommRemoteOffset_f16"
    assert helper_name in funcs, f"Expected @{helper_name} in module, got {list(funcs)}"
    helper = funcs[helper_name]
    assert f"func.func private @{helper_name}(%ctx: !pto.ptr<i64>, %peer: index) -> index" in helper, helper
    # Helper body: load_scalar reads + arith + divsi + return %delems : index.
    assert helper.count("pto.load_scalar") >= 3, helper  # rankId + 2 window slots
    assert "arith.divsi" in helper
    assert "return %delems : index" in helper, helper
    # Critically, none of the addptr / make_tensor_view forbidden ops appear
    # inside the helper — both must stay at the call site to satisfy
    # PTOAS's same-func constraints (see module docstring).
    assert "pto.addptr" not in helper, "addptr must NOT live in the helper"
    assert "pto.make_tensor_view" not in helper, "make_tensor_view must NOT live in the helper"

    # The kernel calls the helper to get the offset, then emits addptr +
    # make_tensor_view locally so PTOAS sees the addptr→make_tensor_view
    # chain within a single func.func.
    kernel = funcs["kernel"]
    assert f"func.call @{helper_name}(" in kernel
    assert "(!pto.ptr<i64>, index) -> index" in kernel, kernel
    assert "pto.addptr" in kernel, "addptr must live at the call site"
    # The addptr's direct downstream is a make_tensor_view in the same
    # func — that's what makes PTOAS happy.
    addptr_line_idx = next(i for i, line in enumerate(kernel.splitlines()) if "pto.addptr" in line)
    # The next non-trivial line should be a make_tensor_view (allowing one
    # arith.muli in between for the dynamic stride[0] computation).
    following = "\n".join(kernel.splitlines()[addptr_line_idx + 1 : addptr_line_idx + 4])
    assert "pto.make_tensor_view" in following, (
        f"addptr must be followed shortly by make_tensor_view, but next lines were:\n{following}"
    )
    # The local CommContext scalar arithmetic must stay inside the helper.
    assert "pto.load_scalar" not in kernel, "CommContext scalar reads belong in the helper"


def test_one_comm_remote_offset_helper_per_dtype():
    """The module emits a distinct @CommRemoteOffset_<dtype> helper per element dtype."""

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
    # f16 (data) + i32 (signal) — one helper per dtype consumed by a
    # cross-rank op (notify counts; wait stays local-only).
    assert "CommRemoteOffset_f16" in funcs
    assert "CommRemoteOffset_i32" in funcs
    # The element-size constant inside each helper matches the dtype.
    assert "arith.constant 2 : i64" in funcs["CommRemoteOffset_f16"]
    assert "arith.constant 4 : i64" in funcs["CommRemoteOffset_i32"]


def test_remote_load_uses_comm_layout_constants():
    """CommRemoteOffset helper literal offsets equal the comm_layout::k* values."""

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
    helper = funcs["CommRemoteOffset_f16"]

    layout = ir.comm_layout
    rank_idx_unit = layout.RANK_ID_OFFSET // layout.WINDOW_SLOT_STRIDE  # 16 / 8 = 2
    win_idx_unit = layout.WINDOWS_IN_OFFSET // layout.WINDOW_SLOT_STRIDE  # 32 / 8 = 4

    # The helper scaffolding references the rank-slot offset and the
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


def test_get_comm_ctx_emits_no_mlir_aliases_ctx_arg():
    """``pld.system.get_comm_ctx(dist_t)`` is a pure SSA alias.

    The op codegen lambda sets ``current_expr_value`` to the matching
    ``!pto.ptr<i64>`` ctx arg's SSA without emitting any MLIR line. The
    surrounding ``VisitStmt_(AssignStmt)`` then binds the LHS Var to the
    same SSA — so the literal op name must NOT appear in the emitted MLIR.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, data: pld.DistributedTensor[[16, 16], pl.FP32]):
            ctx = pld.system.get_comm_ctx(data)  # noqa: F841 — exercise the alias
            # Touch ``data`` again so it is not DCE'd before the get_comm_ctx call.
            pld.system.wait(data, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Eq)

    mlir = _generate_mlir(P)
    # No literal op name in the emitted MLIR — get_comm_ctx is alias-only.
    assert "pld.system.get_comm_ctx" not in mlir, mlir
    # The ctx ptr arg is still in the func header.
    header = next(line for line in mlir.splitlines() if "func.func @kernel" in line)
    assert "!pto.ptr<i64>" in header, header


def test_rank_emits_pto_load_scalar_at_slot_2_plus_trunci():
    """``pld.system.rank(ctx) -> i32`` reads slot 2 (= kRankIdOffset /
    kWindowSlotStride = 16/8) of the ctx ptr then truncates to i32.

    Asserts that the emitted MLIR contains ``pto.load_scalar %argN[%cK] :
    !pto.ptr<i64> -> i64`` and ``arith.trunci`` — no ``arith.shrui`` (that
    is the nranks path).
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, data: pld.DistributedTensor[[16, 16], pl.FP32]):
            ctx = pld.system.get_comm_ctx(data)
            _r = pld.system.rank(ctx)  # noqa: F841 — exercise rank-only path
            pld.system.wait(data, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Eq)

    mlir = _generate_mlir(P)
    body = mlir.split("func.func @kernel", 1)[1]
    # rank lowering line.
    assert "pto.load_scalar" in body and "!pto.ptr<i64> -> i64" in body, body
    assert "arith.trunci" in body, body
    # rank does not shrui — only nranks does.
    assert "arith.shrui" not in body, body


def test_nranks_emits_pto_load_scalar_plus_shrui_32_plus_trunci():
    """``pld.system.nranks(ctx) -> i32`` reads the SAME slot 2 then
    ``arith.shrui ..., 32`` (high 32 bits = rankNum) then ``arith.trunci``.

    Uses the static_asserted invariant ``kRankNumOffset == kRankIdOffset
    + 4`` (see include/pypto/codegen/distributed/comm_layout.h) to fold
    the rankNum read into the same slot as rankId, saving one load.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, data: pld.DistributedTensor[[16, 16], pl.FP32]):
            ctx = pld.system.get_comm_ctx(data)
            _n = pld.system.nranks(ctx)  # noqa: F841 — exercise nranks
            pld.system.wait(data, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Eq)

    mlir = _generate_mlir(P)
    body = mlir.split("func.func @kernel", 1)[1]
    # nranks lowering: pto.load_scalar + arith.shrui + arith.trunci.
    assert "pto.load_scalar" in body and "!pto.ptr<i64> -> i64" in body, body
    assert "arith.shrui" in body, body
    assert "arith.trunci" in body, body


def test_put_emits_comm_tput_with_attr_and_staging_tile():
    """put codegen emits pto.comm.tput with #pto<atomic_type …> attr + an IR-allocated VEC staging tile."""

    @pl.program
    class PNone:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[16, 64], pl.FP16],
            src: pld.DistributedTensor[[16, 64], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.tensor.put(dst, peer=peer, src=src, atomic=pld.AtomicType.None_)

    mlir = _generate_mlir(PNone)
    tput_line = next(line for line in mlir.splitlines() if "pto.comm.tput(" in line)
    # Plain-store combine mode.
    assert "#pto<atomic_type atomic_none>" in tput_line
    # dst (peer-addressed) and src (local) full-slice partition views, same type.
    assert tput_line.count("!pto.partition_tensor_view<16x64xf16>") == 2
    # A VEC staging tile_buf is materialised in IR (via tile.create) and threaded through buf(...).
    assert "buf(" in tput_line
    assert "!pto.tile_buf<loc=vec" in mlir
    # The staging tile must carry an explicit UB address — PTOAS level3 requires
    # PyPTO to do all tile allocation, so the synthesized stage from ConvertTensorToTileOps
    # must flow through AllocateMemoryAddr.
    stage_alloc_line = next(
        line for line in mlir.splitlines() if "pto.alloc_tile" in line and "tput_stage" in line
    )
    assert "addr = " in stage_alloc_line, (
        f"staging tile must have an explicit addr at level3, got: {stage_alloc_line}"
    )
    # dst is peer-addressed (CommRemoteOffset + addptr); src is local (no addptr
    # needed for its own view).
    assert "func.call @CommRemoteOffset_f16" in mlir
    assert "pto.addptr" in mlir
    assert "_peer_pview" in mlir
    assert "_local_pview" in mlir


def test_put_atomic_add_variant():
    """put with AtomicType.Add lowers to the atomic_add combine attr."""

    @pl.program
    class PAdd:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[128], pl.FP32],
            src: pld.DistributedTensor[[128], pl.FP32],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.tensor.put(dst, peer=peer, src=src, atomic=pld.AtomicType.Add)

    mlir_add = _generate_mlir(PAdd)
    assert "#pto<atomic_type atomic_add>" in mlir_add
    # 1-D [128] transfer flattens to a 1x128 VEC staging tile.
    assert "!pto.partition_tensor_view<128xf32>" in mlir_add


def test_get_emits_comm_tget_with_staging_tile():
    """get codegen emits pto.comm.tget with a VEC staging tile."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[16, 64], pl.FP16],
            src: pld.DistributedTensor[[16, 64], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.tensor.get(dst, peer=peer, src=src)

    mlir = _generate_mlir(P)
    tget_line = next(line for line in mlir.splitlines() if "pto.comm.tget(" in line)
    # dst (local) and src (peer-addressed) full-slice partition views, same type.
    assert tget_line.count("!pto.partition_tensor_view<16x64xf16>") == 2
    # A VEC staging tile_buf is synthesised and threaded through buf(...).
    assert "buf(" in tget_line
    assert "!pto.tile_buf<loc=vec" in mlir
    # src is peer-addressed (CommRemoteOffset + addptr); dst is local.
    assert "func.call @CommRemoteOffset_f16" in mlir
    assert "pto.addptr" in mlir
    assert "_peer_pview" in mlir
    assert "_local_pview" in mlir


def test_get_rank1_transfer_uses_full_slice_partition_view():
    """get on a rank-1 tensor lowers to a full 1-D partition view."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[128], pl.FP32],
            src: pld.DistributedTensor[[128], pl.FP32],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.tensor.get(dst, peer=peer, src=src)

    mlir = _generate_mlir(P)
    assert "pto.comm.tget(" in mlir
    assert "!pto.partition_tensor_view<128xf32>" in mlir
    assert "func.call @CommRemoteOffset_f32" in mlir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
