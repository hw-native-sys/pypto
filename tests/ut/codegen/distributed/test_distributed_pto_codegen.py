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
- Peer addressing is emitted **inline** in the caller's own ``func.func``:
  the CommContext reads plus the byte→element division that yield the
  delta between the local rank's window slice and the peer's slice, then
  ``pto.addptr`` + ``pto.make_tensor_view``. There is no module-level
  ``@CommRemoteOffset_<dtype>`` helper and no ``func.call`` — a mixed
  cube+vector kernel group is one MLIR module holding both the AIC and
  the AIV function, and a helper carrying no ``pto.kernel_kind`` strands
  its value-returning ``return`` outside PTOAS's ``__DAV_VEC__`` section
  guard, breaking the cube compile.
- ``pto.addptr`` and ``pto.make_tensor_view`` MUST live at the call site
  regardless: PTOAS verifies per-function that ``addptr`` directly feeds
  ``make_tensor_view`` / ``initialize_l2g2l_pipe(gm_addr)`` /
  ``load|store_scalar``, AND ``make_tensor_view`` lowers to a strided
  memref whose layout cannot be encoded in a ``!pto.tensor_view<…>``
  return type — so the view could not be returned across a func boundary
  either.
- The inline arithmetic's byte-offset literals are pinned to the constants
  in ``include/pypto/codegen/distributed/comm_layout.h``.
- ``pto.tload`` (remote_load), ``pto.comm.tnotify`` (notify) and
  ``pto.comm.twait`` (wait) consume the partition views with the PTOAS
  attribute spellings (``notifyOp = #pto<notify_op …>`` and
  ``cmp = #pto<wait_cmp …>``).
"""

import re

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from _pto_loc_common import strip_loc
from pypto import DataType, backend, codegen, ir, passes
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


def test_remote_load_binds_type_only_dynamic_partition_extent():
    """A type-only symbol becomes a runtime argument instead of being rejected.

    Was ``..._rejects_...``: a symbol named only in a valid_shape used to have no
    binding, so codegen refused it. MaterializeValidShapeSymbols now prepends it as
    a Scalar[INDEX] parameter fed from the call site, so the partition extent is a
    real SSA value. The ``shape_anchor`` trick in the next test remains valid; it
    is simply no longer the only way to supply the extent.
    """
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

    mlir = _generate_mlir(P)
    assert "func.func @kernel" in mlir
    # The symbol is a scalar parameter now, so the peer partition reads a real
    # SSA value rather than an empty operand.
    remote_partition = next(
        line for line in mlir.splitlines() if "pto.partition_view" in line and "_peer" in line
    )
    assert re.search(r"sizes = \[%c1_index, %[A-Za-z0-9_.$]+\]", remote_partition), remote_partition
    assert "sizes = [%c1_index, ]" not in remote_partition, remote_partition


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


def test_remote_load_emits_inline_offset_arithmetic_with_addptr_at_call_site():
    """remote_load lowers to inline peer-offset arithmetic + addptr + make_tensor_view at call site."""

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

    # No module-level offset helper, and no call to one: the arithmetic is
    # emitted inline so every line sits inside a kernel_kind-guarded section.
    assert not any(name.startswith("CommRemoteOffset") for name in funcs), (
        f"no module-level offset helper may be emitted, got {list(funcs)}"
    )
    assert "CommRemoteOffset" not in mlir, mlir
    assert "func.call" not in funcs["kernel"], funcs["kernel"]

    # The kernel does the CommContext reads itself, then emits addptr +
    # make_tensor_view locally so PTOAS sees the addptr→make_tensor_view
    # chain within a single func.func.
    kernel = funcs["kernel"]
    # Inline body: load_scalar reads (rankId + 2 window slots) + divsi.
    assert kernel.count("pto.load_scalar") >= 3, kernel
    assert "arith.divsi" in kernel, kernel
    assert "pto.addptr" in kernel, "addptr must live at the call site"
    # The addptr's direct downstream is a make_tensor_view in the same func —
    # that's what makes PTOAS happy.
    addptr_line_idx = next(i for i, line in enumerate(kernel.splitlines()) if "pto.addptr" in line)
    following = "\n".join(kernel.splitlines()[addptr_line_idx + 1 : addptr_line_idx + 4])
    assert "pto.make_tensor_view" in following, (
        f"addptr must be followed shortly by make_tensor_view, but next lines were:\n{following}"
    )


def test_remote_store_emits_tstore_with_partition_view_pattern():
    """remote_store lowers to inline peer-offset arithmetic + addptr +
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
    assert "CommRemoteOffset" not in mlir, mlir
    assert kernel.count("pto.load_scalar") >= 3, kernel
    assert "pto.addptr" in kernel, kernel
    assert "pto.make_tensor_view" in kernel, kernel


def test_remote_store_accepts_nd_tile_with_unit_leading_dims():
    """A `[1, H, W]` tile pushed into a `[1, H, W]` window lowers end-to-end.

    The deducer's push contract runs at authoring time, before FlattenTileNdTo2D
    collapses N-D tiles, so it has to admit leading unit dims — rejecting rank > 2
    outright would refuse a program that compiles to a correct 3-D partition view.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            inp: pl.Tensor[[1, 16, 32], pl.FP16],
            data: pld.DistributedTensor[[1, 16, 32], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            tile = pl.load(inp, [0, 0, 0], [1, 16, 32])
            pld.tile.remote_store(tile, target=data, peer=peer, offsets=[0, 0, 0])

    kernel = _split_module(_generate_mlir(P))["kernel"]
    assert "pto.tstore" in kernel, kernel
    assert "!pto.partition_tensor_view<1x16x32xf16>" in kernel, kernel
    assert "_peer_pview" in kernel, kernel


def test_remote_store_emits_atomic_add_attr():
    """``atomic=AtomicType.Add`` makes the cross-rank push a combine.

    Same ``atomicType`` attr ``tile.store`` already emits for split-K
    accumulation — this is what an all-to-all combine needs to sum every peer's
    contribution in place instead of overwriting it.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[[16, 64], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            tile = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[16, 32])
            pld.tile.remote_store(tile, target=data, peer=peer, offsets=[0, 0], atomic=pld.AtomicType.Add)

    kernel = _split_module(_generate_mlir(P))["kernel"]
    assert "pto.tstore" in kernel, kernel
    assert "{atomicType = #pto<atomic_type atomic_add>}" in kernel, kernel


def test_remote_store_omits_atomic_attr_for_plain_store():
    """A plain push emits no atomicType attr — non-atomic codegen is unchanged."""

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

    kernel = _split_module(_generate_mlir(P))["kernel"]
    assert "pto.tstore" in kernel, kernel
    assert "atomicType" not in kernel, kernel


def test_tensor_remote_store_of_computed_value_emits_tstore_without_tput():
    """A computed value pushed with ``pld.tensor.remote_store`` reaches the peer
    as a single ``pto.tstore`` — no TPUT, no staging tile, no GM round-trip.

    This is the end-to-end shape issue #2349 asked for: before it, the value was
    rejected from both directions and the only way to push it was to store it
    back to global memory and TPUT from there.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 64], pl.FP16],
            data: pld.DistributedTensor[[16, 64], pl.FP16],
            peer: pl.Scalar[pl.INT32],
        ):
            scaled = pl.tensor.add(x, x)
            pld.tensor.remote_store(scaled, data, peer, [0, 0])

    kernel = _split_module(_generate_mlir(P))["kernel"]
    assert "pto.tstore" in kernel, kernel
    assert "_peer_pview" in kernel, kernel
    # Peer address translation still happens, but inline at the call site: the
    # module-level @CommRemoteOffset_<dtype> helper carried no pto.kernel_kind and
    # broke the AIC half of a mixed module, so it was replaced by inline arithmetic.
    assert "CommRemoteOffset" not in kernel, kernel
    assert "pto.addptr" in kernel, kernel
    assert "pto.make_tensor_view" in kernel, kernel
    # The push is a direct tstore of the computed tile: no TPUT bounce buffer.
    assert "pto.comm.tput" not in kernel, kernel
    # ...and the vector add's result feeds it directly rather than being spilled
    # to global memory first.
    assert "pto.vadd" in kernel or "pto.add" in kernel, kernel


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


def test_inline_offset_arithmetic_emits_one_element_size_per_dtype():
    """The inline peer-offset arithmetic divides by each op's own element size."""

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
    # No module-level helper of any dtype survives — the arithmetic is inline.
    assert "CommRemoteOffset" not in mlir, mlir
    kernel = funcs["kernel"]
    # f16 (data) + i32 (signal) — both dtypes are consumed by a cross-rank op
    # (notify counts; wait stays local-only), so both element-size divisors
    # appear in the same function.
    assert "arith.constant 2 : i64" in kernel, kernel
    assert "arith.constant 4 : i64" in kernel, kernel
    # Two remote ops → two independent inline offset computations.
    assert kernel.count("arith.divsi") == 2, kernel


def test_remote_load_uses_comm_layout_constants():
    """Inline peer-offset literal offsets equal the comm_layout::k* values."""

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
    kernel = funcs["kernel"]

    layout = ir.comm_layout
    rank_idx_unit = layout.RANK_ID_OFFSET // layout.WINDOW_SLOT_STRIDE  # 16 / 8 = 2
    win_idx_unit = layout.WINDOWS_IN_OFFSET // layout.WINDOW_SLOT_STRIDE  # 32 / 8 = 4

    # The inline scaffolding references the rank-slot offset and the
    # windowsIn-array base in *u64-units*, derived from comm_layout constants.
    #
    # Pin each constant to its ROLE, not merely its presence in the function:
    # the inline arithmetic emits constants through GetOrEmitConstant, which
    # hoists and dedups them into the function's shared constants section, so a
    # bare `arith.constant 2 : index` may equally be an unrelated shape or
    # stride. Matching the *uses* keeps the comm_layout pin load-bearing.
    rank_slot_reads = [
        line
        for line in kernel.splitlines()
        if "pto.load_scalar" in line and f"[%c{rank_idx_unit}_index]" in line
    ]
    assert rank_slot_reads, kernel
    assert f"arith.addi %c{win_idx_unit}_index," in kernel, kernel
    # Element-size for FP16 is 2 bytes; the byte-delta is divided by 2 to
    # reach a pto.addptr-compatible element offset.
    divsi_lines = [line for line in kernel.splitlines() if "arith.divsi" in line]
    assert any("%c2_i64" in line for line in divsi_lines), kernel


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


def test_remote_load_peer_view_matches_column_vector_layout():
    """A column-vector peer view uses the same forced-DN metadata as its local view."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[[1, 1], pl.FP32],
            out: pl.Out[pl.Tensor[[8, 1], pl.FP32]],
            peer: pl.Scalar[pl.INT32],
        ):
            tile = pld.tile.remote_load(
                data,
                peer=peer,
                offsets=[0, 0],
                shape=[8, 1],
                valid_shape=[1, 1],
            )
            return pl.store(tile, [0, 0], out)

    mlir = _generate_mlir(P)
    funcs = _split_module(mlir)
    kernel = funcs["kernel"]
    addptr_line = next(line for line in kernel.splitlines() if "pto.addptr %arg0" in line)
    peer_ptr = re.search(r"(%\d+) = pto\.addptr", addptr_line)
    assert peer_ptr is not None, addptr_line
    peer_view_line = next(
        line for line in kernel.splitlines() if f"pto.make_tensor_view {peer_ptr.group(1)}" in line
    )
    assert "shape = [%c1_index, %c1_index]" in peer_view_line, peer_view_line
    assert "strides = [%c1_index, %c1_index]" in peer_view_line, peer_view_line
    assert "{layout = #pto.layout<dn>}" in peer_view_line, peer_view_line


def test_remote_load_peer_view_respects_explicit_nd_column_vector_view():
    """An explicit ND identity view overrides the default column-vector convention."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            data: pld.DistributedTensor[[8, 1], pl.FP32],
            out: pl.Out[pl.Tensor[[8, 1], pl.FP32]],
            peer: pl.Scalar[pl.INT32],
        ):
            viewed: pld.DistributedTensor[
                [8, 1],
                pl.FP32,
                pl.TensorView(stride=[1, 1], layout=pl.TensorLayout.ND),
            ] = pl.tensor.view(data, [8, 1], layout=pl.TensorLayout.ND)
            tile = pld.tile.remote_load(
                viewed,
                peer=peer,
                offsets=[0, 0],
                shape=[8, 1],
                valid_shape=[8, 1],
            )
            return pl.store(tile, [0, 0], out)

    mlir = _generate_mlir(P)
    funcs = _split_module(mlir)
    kernel = funcs["kernel"]
    addptr_line = next(line for line in kernel.splitlines() if "pto.addptr %arg0" in line)
    peer_ptr = re.search(r"(%\d+) = pto\.addptr", addptr_line)
    assert peer_ptr is not None, addptr_line
    peer_view_line = next(
        line for line in kernel.splitlines() if f"pto.make_tensor_view {peer_ptr.group(1)}" in line
    )
    assert "shape = [%c8_index, %c1_index]" in peer_view_line, peer_view_line
    assert "strides = [%c1_index, %c1_index]" in peer_view_line, peer_view_line
    assert "{layout = #pto.layout<nd>}" in peer_view_line, peer_view_line


def test_remote_store_rank3_implicit_column_vector_matches_local_view_strides():
    """Rank-3 implicit column-vector peer and local views use identical DN strides."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            inp: pl.Tensor[[3, 1], pl.FP32],
            data: pld.DistributedTensor[[2, 3, 1], pl.FP32],
            peer: pl.Scalar[pl.INT32],
        ):
            tile = pl.load(inp, [0, 0], [3, 1])
            pld.tile.remote_store(tile, data, peer=peer, offsets=[0, 0, 0])

    mlir = _generate_mlir(P)
    kernel = _split_module(mlir)["kernel"]
    rank3_views = [
        line
        for line in kernel.splitlines()
        if "pto.make_tensor_view" in line and "shape = [%c2_index, %c3_index, %c1_index]" in line
    ]
    assert len(rank3_views) == 2, kernel
    assert all("strides = [%c3_index, %c1_index, %c1_index]" in line for line in rank3_views), rank3_views
    assert all("{layout = #pto.layout<dn>}" in line for line in rank3_views), rank3_views


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
    lines = mlir.splitlines()
    notify_idx = next(i for i, line in enumerate(lines) if "pto.comm.tnotify(" in line)
    assert "pto.barrier <PIPE_ALL>" in lines[notify_idx - 1], (
        f"expected a PIPE_ALL drain immediately before tnotify, got: {lines[notify_idx - 1]}"
    )
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


def test_remote_store_cacheinvalid_fence_before_releasing_notify():
    """A remote_store followed by a notify lowers to a peer-region
    ``pto.cmo.cacheinvalid`` + GM ``pto.fence.barrier_all`` (emitted by the
    remote_store codegen at the peer address), in that order, before the
    ``pto.comm.tnotify`` that releases it (data-before-signal)."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            inp: pl.Tensor[[1, 32], pl.FP32],
            dst: pld.DistributedTensor[[1, 32], pl.FP32],
            signal: pld.DistributedTensor[[16, 16], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            local = pl.load(inp, [0, 0], [1, 32])
            pld.tile.remote_store(local, target=dst, peer=peer, offsets=[0, 0])
            pld.system.notify(signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.Set)

    mlir = _generate_mlir(P)
    lines = mlir.splitlines()
    store_idx = next(i for i, line in enumerate(lines) if "pto.tstore" in line)
    cinv_idx = next(i for i, line in enumerate(lines) if "pto.cmo.cacheinvalid" in line)
    fence_idx = next(i for i, line in enumerate(lines) if "pto.fence.barrier_all" in line)
    tnotify_idx = next(i for i, line in enumerate(lines) if "pto.comm.tnotify(" in line)
    # Order: publishing store -> cacheinvalid -> GM fence -> tnotify.
    assert store_idx < cinv_idx < fence_idx < tnotify_idx, (
        f"expected store({store_idx}) < cacheinvalid({cinv_idx}) < fence({fence_idx}) "
        f"< tnotify({tnotify_idx})"
    )
    assert "#pto.fence_scope<gm>" in lines[fence_idx], lines[fence_idx]
    # Whole-tensor cacheinvalid: the region form addresses the dst via a partition view.
    assert "single_cache_line" in lines[cinv_idx], lines[cinv_idx]


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


def test_wait_casts_loop_induction_expected_to_i32():
    """A pl.range loop induction variable used as ``expected`` is cast to i32.

    ``pl.range``'s induction variable defaults to DataType.INDEX; arithmetic
    on it (``step + 1``) stays INDEX. PTOAS's TWaitOp declares ``cmpValue``
    as AnySignlessInteger with a 32-bit-width verifier check, so an
    uncast ``index`` operand fails to parse ("invalid kind of type
    specified"). Regression for issue #2222.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            signal: pld.DistributedTensor[[16, 16], pl.INT32],
        ):
            for step in pl.range(4):
                pld.system.wait(signal, offsets=[0, 0], expected=step + 1, cmp=pld.WaitCmp.Ge)

    mlir = _generate_mlir(P)
    twait_line = next(line for line in mlir.splitlines() if "pto.comm.twait(" in line)
    assert strip_loc(twait_line).endswith("i32) {cmp = #pto<wait_cmp ge>}"), twait_line
    body = mlir.split("func.func @kernel", 1)[1]
    assert "arith.index_cast" in body and "to i32" in body, body


def test_notify_casts_loop_induction_value_to_i32():
    """A pl.range loop induction variable used as ``value`` is cast to i32.

    Same root cause as ``test_wait_casts_loop_induction_expected_to_i32``,
    for TNotifyOp's ``value`` operand. Regression for issue #2222.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            signal: pld.DistributedTensor[[16, 16], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            for step in pl.range(4):
                pld.system.notify(signal, peer=peer, offsets=[0, 0], value=step + 1, op=pld.NotifyOp.Set)

    mlir = _generate_mlir(P)
    tnotify_line = next(line for line in mlir.splitlines() if "pto.comm.tnotify(" in line)
    assert strip_loc(tnotify_line).endswith("i32) {notifyOp = #pto<notify_op set>}"), tnotify_line
    body = mlir.split("func.func @kernel", 1)[1]
    assert "arith.index_cast" in body and "to i32" in body, body


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
    # dst is peer-addressed (inline peer-offset + addptr); src is local (no
    # addptr needed for its own view).
    assert "CommRemoteOffset" not in mlir, mlir
    # Pin the element-size divisor, not just "some scalar read happened":
    # pto.load_scalar alone is emitted by unrelated lowerings (pld.system.rank,
    # tensor.read), so it would not catch a wrong dtype reaching the inline
    # peer-offset arithmetic. FP16 => 2 bytes.
    assert "arith.constant 2 : i64" in mlir, mlir
    assert "arith.divsi" in mlir, mlir
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
    # After the tput: a tail `pto.barrier <PIPE_ALL>` to drain the DMA pipe (the GM
    # fence does not drain the MTE pipe — without this, atomic/subregion put flakes
    # on device), then the peer-region `pto.cmo.cacheinvalid` + GM
    # `pto.fence.barrier_all` (data-before-signal at the peer address).
    lines = mlir.splitlines()
    tput_idx = next(i for i, line in enumerate(lines) if "pto.comm.tput(" in line)
    assert "pto.barrier <PIPE_ALL>" in lines[tput_idx + 1], lines[tput_idx + 1]
    assert "pto.cmo.cacheinvalid" in lines[tput_idx + 2], lines[tput_idx + 2]
    assert "pto.fence.barrier_all #pto.fence_scope<gm>" in lines[tput_idx + 3], lines[tput_idx + 3]


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


def test_put_atomic_add_bf16_on_ascend910b():
    """A bf16 remote atomic-add is legal on A2/A3 (pto-isa set_atomic_bf16).

    TPUT lands its chunks through the same store pipe as ``tile.store``, so the
    bf16 gate is the shared ``BackendHandler::SupportsBf16AtomicAdd``. The
    Ascend950 rejection lives in the ``AtomicAddDtypeValid`` verifier — see
    ``tests/ut/ir/verifier/test_atomic_add_dtype.py``; here the 910B fixture
    backend must let the put through and emit the combine attr.
    """

    @pl.program
    class PBf16:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[16, 64], pl.BF16],
            src: pld.DistributedTensor[[16, 64], pl.BF16],
            peer: pl.Scalar[pl.INT32],
        ):
            pld.tensor.put(dst, peer=peer, src=src, atomic=pld.AtomicType.Add)

    mlir = _generate_mlir(PBf16)
    tput_line = next(line for line in mlir.splitlines() if "pto.comm.tput(" in line)
    assert "#pto<atomic_type atomic_add>" in tput_line
    assert tput_line.count("!pto.partition_tensor_view<16x64xbf16>") == 2


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
    # src is peer-addressed (inline peer-offset + addptr); dst is local.
    assert "CommRemoteOffset" not in mlir, mlir
    # Element-size divisor pinned per dtype (FP16 => 2 bytes); see the note in
    # test_put_emits_comm_tput_with_attr_and_staging_tile.
    assert "arith.constant 2 : i64" in mlir, mlir
    assert "arith.divsi" in mlir, mlir
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
    assert "CommRemoteOffset" not in mlir, mlir
    # The suite's only FP32 remote op: pins the 4-byte element-size divisor of
    # the inline peer-offset arithmetic. A wrong dtype reaching
    # EmitCommRemoteOffsetInline would emit 2 here and silently address one full
    # window past the peer's slice.
    assert "arith.constant 4 : i64" in mlir, mlir
    assert "arith.divsi" in mlir, mlir
    assert "pto.addptr" in mlir, mlir


def test_mixed_cube_vector_kernel_emits_no_module_level_offset_helper():
    """A comm op in a *mixed* cube+vector kernel emits no module-level helper.

    This is the configuration the inline lowering exists for, and the only one
    where the old module-level ``@CommRemoteOffset_<dtype>`` helper was fatal:
    ExpandMixedKernel splits the kernel into an AIC and an AIV function that
    ptoas compiles from ONE module into ONE ``.cpp``, compiled once per core.
    The helper carried no ``pto.kernel_kind``, so ptoas emitted its body under
    ``#if defined(__DAV_VEC__)`` while leaving the value-returning ``return``
    outside the guard — and the cube compile failed on undeclared identifiers.

    Every other test in this file uses a single InCore function, which never
    produces a second kernel_kind and so cannot catch a regression here.

    What this test does NOT model is comm placement. With no ``pl.split_aiv``
    region, ``pld.system.notify`` classifies SHARED and ExpandMixedKernel copies
    it onto BOTH lanes — the very duplication a region exists to prevent. That
    is deliberate here: pinning the comm phase needs a region, which cannot be
    written inside a function declared ``pl.FunctionType.InCore``, and which,
    written in a plain ``@pl.function``, outlines the comm phase into its own
    AIV function and so dissolves the single mixed module this test is about.
    The kernel is a codegen fixture for the offset lowering, not an authoring
    example; ``test_split_aiv_region_keeps_notify_off_the_cube_lane`` below is
    where placement is under test.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[16, 256], pl.BF16],
            w: pl.Tensor[[256, 256], pl.BF16],
            out: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
            win: pld.DistributedTensor[[16, 256], pl.FP32],
            sig: pld.DistributedTensor[[4, 4], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            acc = pl.matmul(a, w, b_trans=True, out_dtype=pl.FP32)
            out[0:16, 0:256] = acc
            pld.tensor.put(
                dst=win,
                peer=peer,
                src=out,
                dst_offsets=[0, 0],
                src_offsets=[0, 0],
                shape=[16, 256],
            )
            pld.system.notify(target=sig, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.Set)

    optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(P)
    members = [f for f in optimized.functions.values() if ir.is_incore_type(f.func_type)]
    kinds = {str(f.func_type) for f in members}
    assert any("AIC" in k for k in kinds) and any("AIV" in k for k in kinds), (
        f"expected the kernel to split into AIC + AIV, got {sorted(kinds)}"
    )

    # One module carrying both kernel_kinds — exactly what ptoas turns into a
    # single per-core translation unit.
    grouped = ir.Program(members, "kernel", optimized.span)
    mlir = codegen.PTOCodegen().generate(grouped)

    funcs = _split_module(mlir)
    assert not any(name.startswith("CommRemoteOffset") for name in funcs), sorted(funcs)
    assert "CommRemoteOffset" not in mlir, mlir
    assert "func.call" not in mlir, mlir
    # The comm ops are still lowered — a vacuous pass would satisfy the above.
    assert "pto.comm.tput(" in mlir, mlir
    assert "pto.comm.tnotify(" in mlir, mlir
    assert "arith.divsi" in mlir, mlir


def test_split_aiv_region_keeps_notify_off_the_cube_lane():
    """A region pins ``pld.system.notify`` to the AIV function of a MIXED kernel.

    ``pld.system.notify`` is core-agnostic by ISA, so it classifies SHARED, and
    ExpandMixedKernel copies SHARED statements onto BOTH functions of a kernel
    it splits. On the cube lane that is a real hazard: the AIC copy can publish
    the signal before the AIV lane's TPUT has landed the data the signal
    releases, so the peer reads stale bytes. Writing the comm phase inside a
    ``pl.split_aiv`` region is what prevents it — LowerAutoVectorSplit stamps
    the region's no-duplicate calls with ``core_placement="aiv"`` and
    ClassifyCallAffinity resolves that to VECTOR.

    The kernel must be GENUINELY mixed at the InCore level for this to be under
    test at all: the ``pl.at(level=pl.Level.CORE_GROUP)`` block holds both the
    cube matmul and the region, so it outlines into ONE InCore function that
    ExpandMixedKernel really does split into an ``_aic`` / ``_aiv`` pair. (An
    earlier version of this test put the matmul at ``@pl.function`` level, where
    it stays a tensor-level op and only the region is outlined — leaving a
    single AIV function, no cube lane, and nothing for the stamp to do. It
    passed identically with the stamp removed.)

    The discriminating assertion is the pair: notify present in the AIV
    function, ABSENT from the AIC one. With the stamp neutered the notify
    appears in both, and the second assertion fails.

    Property verification stays ON; only the ambient print->parse roundtrip
    instrument is dropped, for a pre-existing asymmetry that has nothing to do
    with placement: the printer re-synthesises scope wrappers around a lowered
    region, so print->parse is not structurally equal here. Suppressing the
    whole context would also drop verification, which is the part that must keep
    running.
    """

    @pl.program
    class P:
        @pl.function
        def kernel(
            self,
            a: pl.Tensor[[16, 64], pl.BF16],
            w: pl.Tensor[[64, 256], pl.BF16],
            out: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
            win: pld.DistributedTensor[[16, 256], pl.FP32],
            sig: pld.DistributedTensor[[4, 4], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                ta = pl.load(a, [0, 0], [16, 64])
                tw = pl.load(w, [0, 0], [64, 256])
                acc = pl.matmul(ta, tw, out_dtype=pl.FP32)
                out = pl.store(acc, [0, 0], out)
                for _aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):  # noqa: B007
                    pld.tensor.put(
                        dst=win,
                        peer=peer,
                        src=out,
                        dst_offsets=[0, 0],
                        src_offsets=[0, 0],
                        shape=[16, 256],
                    )
                    pld.system.notify(
                        target=sig, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd
                    )
            return out

    verify_only: list[passes.PassInstrument] = [
        passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)
    ]
    with passes.PassContext(verify_only):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(P)

    lanes = {
        str(f.func_type): ir.python_print(f)
        for f in optimized.functions.values()
        if ir.is_incore_type(f.func_type)
    }
    aic = [txt for kind, txt in lanes.items() if "AIC" in kind]
    aiv = [txt for kind, txt in lanes.items() if "AIV" in kind]
    # The kernel really was split — otherwise the assertions below are vacuous.
    assert len(aic) == 1 and len(aiv) == 1, sorted(lanes)
    assert "tile.matmul" in aic[0], aic[0]

    assert "pld.system.notify" in aiv[0], aiv[0]
    assert "pld.system.notify" not in aic[0], aic[0]

    # The transient placement carrier is consumed by ExpandMixedKernel, so it
    # must not survive into the final IR.
    assert "core_placement" not in ir.python_print(optimized)

    incore = [f for f in optimized.functions.values() if ir.is_incore_type(f.func_type)]
    mlir = codegen.PTOCodegen().generate(ir.Program(incore, "kernel", optimized.span))
    assert mlir.count("pto.comm.tnotify(") == 1, mlir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
