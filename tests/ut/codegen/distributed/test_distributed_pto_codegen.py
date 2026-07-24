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

- MaterializeDistTensorCtx adds one explicit CommContext IR parameter per
  ``DistributedTensor`` IR param; PTO codegen lowers each one to
  ``!pto.ptr<i64>``.
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

import re

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import DataType, backend, codegen, ir
from pypto.backend import BackendType
from pypto.ir.builder import IRBuilder
from pypto.ir.op.distributed import system_ops as dist_system
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


def test_ctx_arg_materialized_per_distributed_tensor():
    """One explicit ``!pto.ptr<i64>`` arg is emitted per DistributedTensor param."""

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
    # Args after user scalars are the explicit ctx ptrs materialized in IR.
    assert "%arg4: !pto.ptr<i64>" in header, header
    assert "%arg5: !pto.ptr<i64>" in header, header
    # The CtxArg type only appears in the func header at this point (later
    # body uses bind to %argK references). Two DistributedTensors → two ptr
    # declarations.
    assert header.count("!pto.ptr<i64>") == 2, header


def test_remote_load_ragged_tail_partitions_only_valid_extent():
    """A fixed physical remote tile must not read beyond its valid tail."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[[1, 17], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 17], pl.FP32]],
            peer: pl.Scalar[pl.INT32],
        ):
            tile = pld.tile.remote_load(
                data,
                peer=peer,
                offsets=[0, 0],
                shape=[1, 8192],
                valid_shape=[1, 17],
            )
            return pl.store(tile, [0, 0], out)

    mlir = _generate_mlir(P)
    remote_partition = next(
        line for line in mlir.splitlines() if "pto.partition_view" in line and "_peer" in line
    )
    assert "sizes = [%c1_index, %c17_index]" in remote_partition, remote_partition
    alloc = next(line for line in mlir.splitlines() if "pto.alloc_tile" in line)
    assert "valid_row = %c1_index" in alloc, alloc
    assert "valid_col = %c17_index" in alloc, alloc


def test_remote_load_intersects_requested_and_source_valid_extent():
    """remote_load must not partition beyond the source's valid region."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[
                [1, 16],
                pl.FP32,
                pl.TensorView(valid_shape=[1, 8], stride=[], layout=pl.TensorLayout.ND),
            ],
            out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
            peer: pl.Scalar[pl.INT32],
        ):
            tile = pld.tile.remote_load(
                data,
                peer=peer,
                offsets=[0, 0],
                shape=[1, 16],
                valid_shape=[1, 16],
            )
            return pl.store(tile, [0, 0], out)

    mlir = _generate_mlir(P)
    remote_partition = next(
        line for line in mlir.splitlines() if "pto.partition_view" in line and "_peer" in line
    )
    assert "sizes = [%c1_index, %c8_index]" in remote_partition, remote_partition
    alloc = next(line for line in mlir.splitlines() if "pto.alloc_tile" in line)
    assert "valid_row = %c1_index" in alloc, alloc
    assert "valid_col = %c8_index" in alloc, alloc


def test_remote_load_without_valid_shape_uses_source_valid_extent():
    """The four-argument form still partitions only the source's real data."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[
                [1, 16],
                pl.FP32,
                pl.TensorView(valid_shape=[1, 8], stride=[], layout=pl.TensorLayout.ND),
            ],
            out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
            peer: pl.Scalar[pl.INT32],
        ):
            tile = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, 16])
            return pl.store(tile, [0, 0], out)

    mlir = _generate_mlir(P)
    remote_partition = next(
        line for line in mlir.splitlines() if "pto.partition_view" in line and "_peer" in line
    )
    assert "sizes = [%c1_index, %c8_index]" in remote_partition, remote_partition


def test_remote_load_rejects_type_only_dynamic_partition_extent():
    """A repeated type-only symbol is not a runtime codegen binding."""
    n = pl.dynamic("REMOTE_VALID_N")

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[
                [1, 16],
                pl.FP32,
                pl.TensorView(valid_shape=[1, n], stride=[], layout=pl.TensorLayout.ND),
            ],
            out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
            peer: pl.Scalar[pl.INT32],
        ):
            tile = pld.tile.remote_load(
                data,
                peer=peer,
                offsets=[0, 0],
                shape=[1, 16],
                valid_shape=[1, n],
            )
            return pl.store(tile, [0, 0], out)

    with pytest.raises(Exception, match="depends on unbound symbol 'REMOTE_VALID_N'"):
        _generate_mlir(P)


def test_remote_load_intersects_runtime_bound_dynamic_source_valid_extent():
    """A source-valid symbol bound by a tensor shape narrows the partition."""
    n = pl.dynamic("REMOTE_PHYSICAL_N")

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[
                [1, 16],
                pl.FP32,
                pl.TensorView(valid_shape=[1, n], stride=[], layout=pl.TensorLayout.ND),
            ],
            shape_anchor: pl.Tensor[[1, n], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
            peer: pl.Scalar[pl.INT32],
        ):
            tile = pld.tile.remote_load(
                data,
                peer=peer,
                offsets=[0, 0],
                shape=[1, 16],
                valid_shape=[1, 16],
            )
            return pl.store(tile, [0, 0], out)

    mlir = _generate_mlir(P)
    assert "func.func @kernel" in mlir
    assert "REMOTE_PHYSICAL_N" not in mlir
    remote_partition = next(
        line for line in mlir.splitlines() if "pto.partition_view" in line and "_peer" in line
    )
    assert "sizes = [%c1_index, %c16_index]" not in remote_partition, remote_partition
    assert re.search(r"sizes = \[%c1_index, %[A-Za-z0-9_.$]+\]", remote_partition), remote_partition


def test_remote_load_clamps_runtime_bound_fully_valid_dynamic_source():
    """Both call forms clamp a fully-valid dynamic source to its runtime extent."""
    n = pl.dynamic("REMOTE_FULLY_VALID_N")

    @pl.program
    class DefaultCall:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[[1, n], pl.FP32],
            shape_anchor: pl.Tensor[[1, n], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
            peer: pl.Scalar[pl.INT32],
        ):
            tile = pld.tile.remote_load(
                data,
                peer=peer,
                offsets=[0, 0],
                shape=[1, 16],
            )
            return pl.store(tile, [0, 0], out)

    @pl.program
    class ExplicitCall:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[[1, n], pl.FP32],
            shape_anchor: pl.Tensor[[1, n], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
            peer: pl.Scalar[pl.INT32],
        ):
            tile = pld.tile.remote_load(
                data,
                peer=peer,
                offsets=[0, 0],
                shape=[1, 16],
                valid_shape=[1, 16],
            )
            return pl.store(tile, [0, 0], out)

    for program in (DefaultCall, ExplicitCall):
        mlir = _generate_mlir(program)
        min_results = {
            match.group(1)
            for line in mlir.splitlines()
            if (match := re.match(r"\s*(%[A-Za-z0-9_.$]+) = arith\.minsi ", line))
        }
        remote_partitions = [
            line for line in mlir.splitlines() if "pto.partition_view" in line and "_peer" in line
        ]
        assert len(remote_partitions) == 1
        remote_partition = remote_partitions[0]
        assert "sizes = [%c1_index, %c16_index]" not in remote_partition, remote_partition
        size_match = re.search(r"sizes = \[%c1_index, (%[A-Za-z0-9_.$]+)\]", remote_partition)
        assert size_match is not None, remote_partition
        assert size_match.group(1) in min_results, remote_partition


def test_remote_load_codegen_uses_runtime_min_of_source_and_requested_valid_extents():
    """The peer partition consumes the exact min of two bound symbolic extents."""
    source_cols = pl.dynamic("REMOTE_SOURCE_COLS")

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[
                [1, 16],
                pl.FP32,
                pl.TensorView(valid_shape=[1, source_cols], stride=[], layout=pl.TensorLayout.ND),
            ],
            shape_anchor: pl.Tensor[[1, source_cols], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
            peer: pl.Scalar[pl.INT32],
            requested_cols: pl.Scalar[pl.INDEX],
        ):
            tile = pld.tile.remote_load(
                data,
                peer=peer,
                offsets=[0, 0],
                shape=[1, 16],
                valid_shape=[1, requested_cols],
            )
            return pl.store(tile, [0, 0], out)

    class RemoteLoadCollector(ir.IRVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[ir.Call] = []

        def visit_call(self, op: ir.Call) -> None:
            if op.op.name == ir.get_op("pld.tile.remote_load").name:
                self.calls.append(op)
            super().visit_call(op)

    collector = RemoteLoadCollector()
    collector.visit_program(P)
    assert len(collector.calls) == 1
    inferred_type = collector.calls[0].type
    assert isinstance(inferred_type, ir.TileType)
    assert inferred_type.tile_view is not None
    valid_cols = inferred_type.tile_view.valid_shape[1]
    assert isinstance(valid_cols, ir.Min)

    def collect_min_vars(expr: ir.Expr) -> set[str]:
        if isinstance(expr, ir.Var):
            return {expr.name_hint}
        if isinstance(expr, ir.Min):
            return collect_min_vars(expr.left) | collect_min_vars(expr.right)
        return set()

    assert collect_min_vars(valid_cols) == {"REMOTE_SOURCE_COLS", "requested_cols"}

    mlir = _generate_mlir(P)
    min_dependencies = {
        match.group(1): (match.group(2), match.group(3))
        for line in mlir.splitlines()
        if (
            match := re.match(
                r"\s*(%[A-Za-z0-9_.$]+) = arith\.minsi "
                r"(%[A-Za-z0-9_.$]+), (%[A-Za-z0-9_.$]+) : index",
                line,
            )
        )
    }
    assert min_dependencies, mlir
    remote_partition = next(
        line for line in mlir.splitlines() if "pto.partition_view" in line and "_peer" in line
    )
    size_match = re.search(r"sizes = \[%c1_index, (%[A-Za-z0-9_.$]+)\]", remote_partition)
    assert size_match is not None, remote_partition

    def collect_argument_dependencies(value: str) -> set[str]:
        if re.fullmatch(r"%arg[0-9]+", value):
            return {value}
        if value in min_dependencies:
            left, right = min_dependencies[value]
            return collect_argument_dependencies(left) | collect_argument_dependencies(right)
        return set()

    partition_size = size_match.group(1)
    assert partition_size in min_dependencies, remote_partition
    assert len(collect_argument_dependencies(partition_size)) == 2, remote_partition


def test_remote_load_accepts_scalar_bound_dynamic_partition_extent():
    """An explicit scalar valid extent is already bound in the kernel."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[[1, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
            peer: pl.Scalar[pl.INT32],
            valid_cols: pl.Scalar[pl.INT32],
        ):
            tile = pld.tile.remote_load(
                data,
                peer=peer,
                offsets=[0, 0],
                shape=[1, 16],
                valid_shape=[1, valid_cols],
            )
            return pl.store(tile, [0, 0], out)

    mlir = _generate_mlir(P)
    remote_partition = next(
        line for line in mlir.splitlines() if "pto.partition_view" in line and "_peer" in line
    )
    assert re.search(r"sizes = \[%c1_index, %[A-Za-z0-9_.$]+\]", remote_partition), remote_partition
    index_casts = re.findall(r"arith\.index_cast %[A-Za-z0-9_.$]+ : i32 to index", mlir)
    assert len(index_casts) >= 2, mlir  # peer plus valid_cols


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


def test_remote_store_emits_tstore_with_partition_view_pattern():
    """remote_store lowers to func.call @CommRemoteOffset_<dtype> + addptr +
    make_tensor_view + partition_view + pto.tstore at the call site."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[[16, 64], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            tile = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[16, 32])
            pld.tile.remote_store(tile, target=data, peer=peer, offsets=[0, 0])

    mlir = _generate_mlir(P)
    funcs = _split_module(mlir)
    kernel = funcs["kernel"]

    # The 2-D tile partition view type for the store side carries the tile's
    # height×width (16×32) and the target's dtype.
    assert "!pto.partition_tensor_view<16x32xf16>" in kernel, kernel
    # tstore uses the peer-addressed partition_view, naming the peer view per
    # the EmitPartitionViewPTO contract.
    assert "pto.tstore" in kernel, kernel
    assert "_peer_pview" in kernel, kernel
    # Address translation lives at the call site (same constraints as remote_load).
    assert "func.call @CommRemoteOffset_f16" in kernel, kernel
    assert "pto.addptr" in kernel, kernel
    assert "pto.make_tensor_view" in kernel, kernel


def test_remote_store_pads_partition_view_with_ones_for_3d_target():
    """For an N-D (N > 2) target, the partition_view rank matches the target
    rank — leading dims are size-1 (matching notify's one_dims(rank, "1")
    pattern) so the 2-D tile lands on the inner two dims of the peer slice
    without forcing the caller to reshape.

    This is the regression guard for the previous hidden bug where a 3-D
    DistributedTensor target passed the verifier (target_rank > 0) but the
    codegen emitted a rank-mismatched ``pto.partition_view`` that PTOAS
    would reject.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            inp: pl.Tensor[[16, 32], pl.FP16],
            data: pld.DistributedTensor[[4, 16, 64], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            tile = pl.load(inp, [0, 0], [16, 32])
            pld.tile.remote_store(tile, target=data, peer=peer, offsets=[0, 0, 0])

    mlir = _generate_mlir(P)
    funcs = _split_module(mlir)
    kernel = funcs["kernel"]

    # The partition view on the store side must be 3-D, with a leading 1 in
    # the outermost dim and the tile's two inner dims appended.
    assert "!pto.partition_tensor_view<1x16x32xf16>" in kernel, kernel
    assert "pto.tstore" in kernel, kernel


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


def test_remote_load_peer_view_preserves_explicit_tensor_view_layout_and_strides():
    """remote_load reuses explicit TensorView metadata for the peer view."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[[4, 8], pl.FP32],
            out: pl.Tensor[[8, 4], pl.FP32],
            peer: pl.Scalar[pl.INT32],
        ):
            viewed: pld.DistributedTensor[
                [8, 4],
                pl.FP32,
                pl.TensorView(stride=[1, 8], layout=pl.TensorLayout.DN),
            ] = pl.tensor.view(data, [8, 4], layout=pl.TensorLayout.DN)
            t = pld.tile.remote_load(viewed, peer=peer, offsets=[0, 0], shape=[8, 4])
            pl.store(t, [0, 0], out)

    mlir = _generate_mlir(P)
    funcs = _split_module(mlir)
    kernel = funcs["kernel"]
    addptr_line = next(line for line in kernel.splitlines() if "pto.addptr %arg0" in line)
    peer_ptr = re.search(r"(%\d+) = pto\.addptr", addptr_line)
    assert peer_ptr is not None, addptr_line
    peer_view_line = next(
        line for line in kernel.splitlines() if f"pto.make_tensor_view {peer_ptr.group(1)}" in line
    )
    assert "shape = [%c8_index, %c4_index]" in peer_view_line, peer_view_line
    assert "strides = [%c1_index, %c8_index]" in peer_view_line, peer_view_line
    assert "{layout = #pto.layout<dn>}" in peer_view_line, peer_view_line


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


def test_plain_distributed_alias_preserves_comm_ctx():
    """A direct AssignStmt alias keeps the source view, base pointer, and ctx."""
    ty = ir.DistributedTensorType([16, 16], DataType.INT32)

    ib = IRBuilder()
    with ib.function("alias_wait", type=ir.FunctionType.InCore) as f:
        data = f.param("data", ty)
        f.param("data_ctx", ir.CommCtxType.get())
        alias = ib.let("alias", data)
        ib.eval_stmt(dist_system.wait(alias, [0, 0], 1, ir.WaitCmp.Eq))
        ib.return_stmt()

    program = ir.Program([f.get_result()], "alias_wait", ir.Span.unknown())
    mlir = codegen.PTOCodegen().generate(program)
    body = mlir.split("func.func @alias_wait", 1)[1]
    assert "pto.comm.twait" in body, body


def test_tensor_view_preserves_loop_carried_distributed_metadata():
    """Post-loop views keep the distributed tensor's base pointer and ctx."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, data: pld.DistributedTensor[[16, 16], pl.INT32]):
            for _i, (carried,) in pl.range(1, init_values=(data,)):
                result = pl.yield_(carried)
            viewed = pl.tensor.view(result, [16, 16])
            pld.system.wait(viewed, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Eq)

    mlir = _generate_mlir(P)
    body = mlir.split("func.func @kernel", 1)[1]
    assert body.count("pto.make_tensor_view %arg0") >= 2, body
    assert "pto.comm.twait" in body, body


def test_tensor_view_preserves_while_carried_distributed_metadata():
    """The while-loop return alias keeps the distributed base pointer and ctx."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, data: pld.DistributedTensor[[16, 16], pl.INT32]):
            limit: pl.Scalar[pl.INT64] = 1
            for (carried,) in pl.while_(init_values=(data,)):
                pl.cond(limit > 0)
                result = pl.yield_(carried)
            viewed = pl.tensor.view(result, [16, 16])
            pld.system.wait(viewed, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Eq)

    mlir = _generate_mlir(P)
    body = mlir.split("func.func @kernel", 1)[1]
    assert body.count("pto.make_tensor_view %arg0") >= 2, body
    assert "pto.comm.twait" in body, body


def test_tensor_view_preserves_if_merged_distributed_metadata():
    """A distributed tensor merged by an if keeps its base pointer and ctx."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[[16, 16], pl.INT32],
            cond: pl.Scalar[pl.BOOL],
        ):
            result = data
            if cond:
                result = data
            viewed = pl.tensor.view(result, [16, 16])
            pld.system.wait(viewed, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Eq)

    mlir = _generate_mlir(P)
    body = mlir.split("func.func @kernel", 1)[1]
    assert "scf.if" in body, body
    assert body.count("pto.make_tensor_view %arg0") >= 2, body
    assert "pto.comm.twait" in body, body


def test_if_merged_distributed_metadata_rejects_conflicting_contexts():
    """An in-place if cannot select data and context from different allocations."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            lhs: pld.DistributedTensor[[16, 16], pl.INT32],
            rhs: pld.DistributedTensor[[16, 16], pl.INT32],
            cond: pl.Scalar[pl.BOOL],
        ):
            result = lhs
            if cond:
                result = rhs
            pld.system.wait(result, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Eq)

    with pytest.raises(
        ValueError,
        match="Assigning a different DistributedTensor in each branch of an `if` is not supported",
    ):
        _generate_mlir(P)


def test_rank_emits_pto_load_scalar_at_slot_2_plus_trunci():
    """``pld.system.rank(ctx)`` reads slot 2 (= kRankIdOffset /
    kWindowSlotStride = 16/8) then truncates to signless ``i32`` for PTOAS.

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
    assert "arith.trunci" in body and "to i32" in body, body
    assert "to ui32" not in body, body
    # rank does not shrui — only nranks does.
    assert "arith.shrui" not in body, body


def test_nranks_emits_pto_load_scalar_plus_shrui_32_plus_trunci():
    """``pld.system.nranks(ctx)`` reads the SAME slot 2 then
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
    assert "arith.trunci" in body and "to i32" in body, body
    assert "to ui32" not in body, body


def test_rank_var_reuse_no_ui32_in_notify_and_compare():
    """``pld.rank`` SSA stays signless ``i32`` when reused in compare and notify offsets.

    Mirrors ``test_l3_allreduce`` ``reduce_step`` barrier pattern: without this,
    ``EmitCastToIndex`` / ``VisitCmpExpr`` treat IR unsigned scalars as ``ui32`` while
    rank lowering defines the var as ``i32``, and PTOAS rejects mixed uses.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[[16, 16], pl.FP32],
            signal: pld.DistributedTensor[[2, 1], pl.INT32],
        ):
            ctx = pld.system.get_comm_ctx(data)
            my_rank = pld.system.rank(ctx)
            for peer in pl.range(2):
                if peer != my_rank:
                    pld.system.notify(
                        signal,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )

    mlir = _generate_mlir(P)
    body = mlir.split("func.func @kernel", 1)[1]
    assert "arith.trunci" in body and "to i32" in body, body
    assert "ui32" not in body, body
    assert "unrealized_conversion_cast" not in body, body
    assert "my_rank" in body, body
    assert "arith.index_cast" in body and "i32 to index" in body, body


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
    # PyPTO to do all tile allocation, so the IR-materialized stage from ConvertTensorToTileOps
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


def test_put_chunk_shrinks_staging_tile_keeping_full_partition_view():
    """``chunk_rows`` / ``chunk_cols`` shrink the VEC staging tile while the
    partition views keep the full transfer extent — pto-isa TPUT then 2-D-slides
    the full transfer through the sub-tile, so transfers larger than UB no longer
    need a full tile."""

    @pl.program
    class PChunk:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[16, 64], pl.FP16],
            src: pld.DistributedTensor[[16, 64], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.tensor.put(dst, peer=peer, src=src, atomic=pld.AtomicType.None_, chunk_rows=4, chunk_cols=32)

    mlir = _generate_mlir(PChunk)
    tput_line = next(line for line in mlir.splitlines() if "pto.comm.tput(" in line)
    # Partition views still describe the FULL 16x64 transfer (TPUT reads the full
    # extent from these and chunks internally).
    assert tput_line.count("!pto.partition_tensor_view<16x64xf16>") == 2
    # The staging tile is the [4, 32] chunk, not the full [16, 64] transfer.
    stage_alloc_line = next(
        line for line in mlir.splitlines() if "pto.alloc_tile" in line and "tput_stage" in line
    )
    assert "rows=4" in stage_alloc_line and "cols=32" in stage_alloc_line, (
        f"staging tile must be the [4, 32] chunk, got: {stage_alloc_line}"
    )
    # A drain barrier is emitted immediately after the tput so a following
    # cross-rank notify can't race the chunked stores (PTOAS#872 workaround).
    lines = mlir.splitlines()
    tput_idx = next(i for i, line in enumerate(lines) if "pto.comm.tput(" in line)
    assert "pto.barrier <PIPE_ALL>" in lines[tput_idx + 1], (
        f"expected a PIPE_ALL drain right after tput, got: {lines[tput_idx + 1]}"
    )


def test_put_pipeline_emits_two_staging_buffers_in_one_buf_group():
    """``pipeline=True`` emits two VEC staging tiles inside a single ``buf(...)``
    operand group, each contributing a trailing ``!pto.tile_buf`` type — pto-isa's
    ping-pong TPUT overload."""

    @pl.program
    class PPipe:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[16, 64], pl.FP16],
            src: pld.DistributedTensor[[16, 64], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.tensor.put(
                dst,
                peer=peer,
                src=src,
                atomic=pld.AtomicType.None_,
                chunk_rows=4,
                chunk_cols=32,
                pipeline=True,
            )

    mlir = _generate_mlir(PPipe)
    tput_line = next(line for line in mlir.splitlines() if "pto.comm.tput(" in line)
    # Both ping/pong tiles ride in a single buf(...) group: two comma-separated
    # SSA tile operands and two trailing tile_buf types.
    buf_inner = tput_line.split("buf(", 1)[1].split(")", 1)[0]
    assert buf_inner.count(",") == 1, f"expected two staging tiles in buf(...), got: {tput_line}"
    assert tput_line.count("!pto.tile_buf<loc=vec") == 2, (
        f"double-buffered tput must list two tile_buf types, got: {tput_line}"
    )
    # Two distinct staging tiles are allocated (ping + pong), each the [4, 32] chunk.
    ping = next(line for line in mlir.splitlines() if "pto.alloc_tile" in line and "tput_stage_ping" in line)
    pong = next(line for line in mlir.splitlines() if "pto.alloc_tile" in line and "tput_stage_pong" in line)
    for line in (ping, pong):
        assert "rows=4" in line and "cols=32" in line, f"staging tile must be [4, 32], got: {line}"


def test_get_pipeline_emits_two_staging_buffers_in_one_buf_group():
    """``pipeline=True`` on get emits the two-buffer ping-pong TGET form."""

    @pl.program
    class PGetPipe:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[16, 64], pl.FP16],
            src: pld.DistributedTensor[[16, 64], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.tensor.get(dst, peer=peer, src=src, chunk_rows=4, chunk_cols=32, pipeline=True)

    mlir = _generate_mlir(PGetPipe)
    tget_line = next(line for line in mlir.splitlines() if "pto.comm.tget(" in line)
    buf_inner = tget_line.split("buf(", 1)[1].split(")", 1)[0]
    assert buf_inner.count(",") == 1, f"expected two staging tiles in buf(...), got: {tget_line}"
    assert tget_line.count("!pto.tile_buf<loc=vec") == 2, (
        f"double-buffered tget must list two tile_buf types, got: {tget_line}"
    )
    assert any("tget_stage_ping" in line for line in mlir.splitlines())
    assert any("tget_stage_pong" in line for line in mlir.splitlines())


def test_put_subregion_dynamic_shape_with_chunk():
    """A dynamic subregion transfer extent emits a dynamic partition view while
    the staging tile stays statically sized from the chunk — pto-isa chunks the
    runtime extent. The fixed window stays static; only the transfer is dynamic."""

    @pl.program
    class PDyn:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[16, 64], pl.FP16],
            src: pld.DistributedTensor[[16, 64], pl.FP16],
            peer: pl.Scalar[pl.INT32],
            n: pl.Scalar[pl.INT32],
        ):
            pld.tensor.put(
                dst,
                peer=peer,
                src=src,
                dst_offsets=[0, 0],
                src_offsets=[0, 0],
                shape=[n, 64],
                chunk_rows=4,
                chunk_cols=32,
            )

    mlir = _generate_mlir(PDyn)
    tput_line = next(line for line in mlir.splitlines() if "pto.comm.tput(" in line)
    # Dynamic rows in the partition view (the `n` runtime extent), static cols.
    assert tput_line.count("!pto.partition_tensor_view<?x64xf16>") == 2, tput_line
    # Staging tile is the static [4, 32] chunk (UB allocation is static).
    stage_alloc_line = next(
        line for line in mlir.splitlines() if "pto.alloc_tile" in line and "tput_stage" in line
    )
    assert "rows=4" in stage_alloc_line and "cols=32" in stage_alloc_line, stage_alloc_line


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


def test_put_subregion_uses_offset_partition_views():
    """offset put lowers dst/src subregions to matching partition views."""

    @pl.program
    class PSubregion:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[16, 64], pl.FP16],
            src: pld.DistributedTensor[[8, 64], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.tensor.put(
                dst,
                peer=peer,
                src=src,
                dst_offsets=[3, 0],
                src_offsets=[1, 0],
                shape=[1, 64],
                atomic=pld.AtomicType.None_,
            )

    mlir = _generate_mlir(PSubregion)
    tput_line = next(line for line in mlir.splitlines() if "pto.comm.tput(" in line)
    assert tput_line.count("!pto.partition_tensor_view<1x64xf16>") == 2
    assert re.search(r"offsets = \[%c3(?:_\w+)?, %c0(?:_\w+)?\]", mlir), mlir
    assert re.search(r"offsets = \[%c1(?:_\w+)?, %c0(?:_\w+)?\]", mlir), mlir
    assert "pto.barrier <PIPE_ALL>" in mlir
    assert "!pto.tile_buf<loc=vec" in mlir


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
    # A VEC staging tile_buf is materialised in IR (via tile.create) and threaded through buf(...).
    assert "buf(" in tget_line
    assert "!pto.tile_buf<loc=vec" in mlir
    stage_alloc_line = next(
        line for line in mlir.splitlines() if "pto.alloc_tile" in line and "tget_stage" in line
    )
    assert "addr = " in stage_alloc_line, (
        f"staging tile must have an explicit addr at level3, got: {stage_alloc_line}"
    )
    # src is peer-addressed (CommRemoteOffset + addptr); dst is local.
    assert "func.call @CommRemoteOffset_f16" in mlir
    assert "pto.addptr" in mlir
    assert "_peer_pview" in mlir
    assert "_local_pview" in mlir
    assert "pto.barrier <PIPE_ALL>" in mlir


def test_get_subregion_uses_offset_partition_views():
    """offset get lowers dst/src subregions to matching partition views."""

    @pl.program
    class PSubregion:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[16, 64], pl.FP16],
            src: pld.DistributedTensor[[8, 64], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.tensor.get(
                dst,
                peer=peer,
                src=src,
                dst_offsets=[3, 0],
                src_offsets=[1, 0],
                shape=[1, 64],
            )

    mlir = _generate_mlir(PSubregion)
    tget_line = next(line for line in mlir.splitlines() if "pto.comm.tget(" in line)
    assert tget_line.count("!pto.partition_tensor_view<1x64xf16>") == 2
    assert re.search(r"offsets = \[%c3(?:_\w+)?, %c0(?:_\w+)?\]", mlir), mlir
    assert re.search(r"offsets = \[%c1(?:_\w+)?, %c0(?:_\w+)?\]", mlir), mlir
    assert "pto.barrier <PIPE_ALL>" in mlir
    assert "!pto.tile_buf<loc=vec" in mlir


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
