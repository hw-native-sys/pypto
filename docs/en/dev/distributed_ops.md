# Distributed Operators (N6)

## Overview

The N6 distributed op family gives the Python DSL direct, typed access to the
hardware's cross-rank communication primitives. Every op operates against a
**window-bound** [`DistributedTensorType`](ir/02-types.md) — a tensor whose
storage is a slice of a symmetric, per-rank communication window allocated by
`pld.alloc_window_buffer`. A plain `TensorType` is *rejected* by every verifier
in this family (strict kind-trait matching — `As<DistributedTensorType>` does
not match a plain `TensorType`), so a non-window-bound tensor can never be fed
into a cross-rank operation by accident.

There are **six ops** and **three ABI enums**:

| Op | Direction | Result | Hardware |
| -- | --------- | ------ | -------- |
| `pld.tile.remote_load` | pull (read peer → local tile) | `TileType` | TLOAD |
| `pld.tile.remote_store` | push (write local tile → peer) | `Unknown` (side effect) | TSTORE |
| `pld.tensor.get` | pull (read peer → local GM) | `Unknown` (side effect) | TGET |
| `pld.tensor.put` | push (write local → peer) | `Unknown` (side effect) | TPUT |
| `pld.system.notify` | signal a peer's slot | `Unknown` (side effect) | TNOTIFY |
| `pld.system.wait` | block on own slot | `Unknown` (side effect) | TWAIT |

The five side-effect-only ops produce [`UnknownType`](ir/02-types.md): they
exist for their cross-rank effect, not for an SSA value a consumer reads.

## Namespacing: why `tile.*` vs `tensor.*` vs `system.*`

The namespace encodes the IR level the op lives at, not an arbitrary grouping:

- **`pld.tile.remote_load`** produces a *tile* (on-core SRAM region), so it is a
  sibling of `tile.load` and lives in `pld.tile`.
- **`pld.tile.remote_store`** consumes a *tile* (the symmetric write companion
  of `remote_load`), so it is a sibling of `tile.store` and lives in
  `pld.tile`.
- **`pld.tensor.get`** reads and writes *tensor* (GM) operands — both `dst` and
  `src` are window-bound `DistributedTensor` views. The VEC staging tile that
  TGET bounces through is materialised by `ConvertTensorToTileOps` as an
  internal `pld.tile.get`, never on the DSL surface. It is therefore a sibling
  of `pld.tensor.alloc_window_buffer` / `pld.tensor.window`, **not** of the
  tile-producing `remote_load`.
- **`pld.tensor.put`** reads and writes *tensor* (GM) operands — both `dst` and
  `src` are window-bound `DistributedTensor` views. The VEC staging tile that
  TPUT bounces through is materialised by `ConvertTensorToTileOps` as an
  internal `pld.tile.put`, never on the DSL surface. It is therefore a sibling
  of `pld.tensor.alloc_window_buffer` / `pld.tensor.window`, **not** of the
  tile-producing `remote_load`.
- **`pld.system.notify` / `pld.system.wait`** drive the per-rank signal slot —
  pure control-plane synchronisation with no data operand — so they live in
  `pld.system`.

## ABI enums (`include/pypto/ir/comm.h`)

The three enums are an **append-only ABI**. Their underlying `int` values are
serialised as the op's kwarg payload (`op` for notify, `cmp` for wait, `atomic`
for put) and cast back to the enum at codegen time. New variants may only be
added **at the end** so existing IR and cached programs keep their meaning.

```cpp
enum class NotifyOp : int { kAtomicAdd = 0, kSet = 1 };   // pld.system.notify
enum class WaitCmp  : int { kEq = 0,        kGe = 1 };     // pld.system.wait
enum class AtomicType : int { kNone = 0,    kAdd = 1 };    // pld.tensor.put
```

| Enum | Variant | Meaning |
| ---- | ------- | ------- |
| `NotifyOp` | `kAtomicAdd` | atomically add `value` into the peer's signal slot |
| `NotifyOp` | `kSet` | non-atomic store of `value` into the peer's signal slot |
| `WaitCmp` | `kEq` | block until `*signal_slot == expected` |
| `WaitCmp` | `kGe` | block until `*signal_slot >= expected` |
| `AtomicType` | `kNone` | plain remote store — overwrite the peer's dst slice |
| `AtomicType` | `kAdd` | atomically add the source data into the peer's dst slice |

Each enum is mirrored across three layers (C++ `enum class` → `nb::enum_` in the
bindings → `.pyi` stub) and surfaced to the DSL as `pld.NotifyOp` /
`pld.WaitCmp` / `pld.AtomicType`. The deducer validates the packed `int`
against the enum range so codegen can cast back without a second guard.

## Op reference

### `pld.tile.remote_load` (TLOAD)

```text
pld.tile.remote_load(target, peer, offsets, shape) -> TileType(shape, target.dtype)
```

Reads a region of the `peer` rank's slice of a window-bound `DistributedTensor`
into a local tile. Mirrors `tile.load` at the IR level (positional `offsets` /
`shape` tuples, `TileType` result) but the source is a *remote* slice — the
address translation is realised at codegen by
`CommRemoteOffset(ctx, peer) + addptr + make_tensor_view`.

Verifier: `target` must be `DistributedTensorType`; `peer` must be a
`ScalarType` rank index; `offsets` / `shape` must each be a `MakeTuple` whose
rank equals `target.shape.size()`.

DSL (`python/pypto/language/distributed/op/tile_ops.py`) exposes `peer` /
`offsets` / `shape` as keyword-only for readability; the IR op keeps them
positional, matching `tile.load`.

### `pld.tile.remote_store` (TSTORE)

```text
pld.tile.remote_store(src_tile, target, peer, offsets) -> Unknown
```

Writes a local tile into a region of the `peer` rank's slice of a window-bound
`DistributedTensor`. Mirrors `tile.store` at the IR level (positional `offsets`
tuple + side-effect-only return) but the destination is a *remote* slice —
address translation happens at codegen via `CommRemoteOffset(ctx, peer) +
addptr + make_tensor_view`.

Verifier: `src_tile` must be `TileType`; `target` must be
`DistributedTensorType`; `peer` must be a `ScalarType` rank index; `offsets`
must be a `MakeTuple` whose rank equals `target.shape.size()`; `src_tile.dtype`
must match `target.dtype`.

Codegen: the tile is 2-D (height × width) after the standard tile pipeline; the
emitted `pto.partition_view` has the same rank as `target`, with the leading
`(target.rank - 2)` dims set to size 1 (matching `notify`'s `one_dims(rank,
"1")` pattern). This lets a 2-D tile push land on the inner two dims of any
N-D peer slice (N ≥ 2) without forcing the caller to reshape — and it is the
regression guard against the older codegen that emitted a fixed-2D
`partition_view` regardless of target rank.

DSL (`python/pypto/language/distributed/op/tile_ops.py`) exposes `target` /
`peer` / `offsets` as keyword-only for readability; the IR op keeps them
positional, matching `tile.store`.

### `pld.tensor.put` (TPUT)

```text
pld.tensor.put(dst, peer, src, *, atomic: int) -> Unknown
pld.tensor.put(dst, peer, src, dst_offsets, src_offsets, shape, *, atomic: int) -> Unknown
```

Synchronously writes the local window-bound `src` into the `peer` rank's slice
of the window-bound `dst`. Both operands are GM-level `DistributedTensor`
views; the VEC staging tile is materialised by `ConvertTensorToTileOps` as an
internal `tile.create + pld.tile.put`, so it flows through PyPTO's memory
allocator and never appears on the DSL surface.

With no offsets/shape this writes the full local `src` slice to the full peer
`dst` slice. Supplying `dst_offsets`, `src_offsets`, and `shape` narrows the
transfer to matching subregions; all three must be provided together.

Verifier: `dst` / `src` must both be `DistributedTensorType`; `peer` must be a
`ScalarType`; `dst` and `src` must share element type, rank, and **positive
static** dimensions. Full-slice `put` requires identical shape; subregion `put`
allows different full slice extents as long as the explicit transfer region is
in bounds. `atomic` selects overwrite vs atomic-add (see `AtomicType`).

### `pld.tensor.get` (TGET)

```text
pld.tensor.get(dst, peer, src) -> Unknown
pld.tensor.get(dst, peer, src, dst_offsets, src_offsets, shape) -> Unknown
```

Synchronously reads the `peer` rank's slice of the window-bound `src` into the
local window-bound `dst`. Both operands are GM-level `DistributedTensor`
views; the VEC staging tile is materialised by `ConvertTensorToTileOps` as an
internal `tile.create + pld.tile.get`, so it flows through PyPTO's memory
allocator and never appears on the DSL surface.

With no offsets/shape this reads the full peer `src` slice into the full local
`dst` slice. Supplying `dst_offsets`, `src_offsets`, and `shape` narrows the
transfer to matching subregions; all three must be provided together.

Verifier: `dst` / `src` must both be `DistributedTensorType`; `peer` must be a
`ScalarType`; `dst` and `src` must share element type, rank, and **positive
static** dimensions. Full-slice `get` requires identical shape; subregion `get`
allows different full slice extents as long as the explicit transfer region is
in bounds. `get` accepts no keyword attributes.

### `pld.system.notify` (TNOTIFY)

```text
pld.system.notify(target, peer, offsets, value, *, op: int) -> Unknown
```

Writes `value` into the `peer` rank's signal slot of `target` (a window-bound
`DistributedTensor`, typically a 1-D INT32 "signal matrix"). `op` selects
atomic-add vs set (see `NotifyOp`).

Verifier: `target` must be `DistributedTensorType`; `peer` and `value` must be
`ScalarType`; `offsets` must be a `MakeTuple` of rank equal to the target rank.

### `pld.system.wait` (TWAIT)

```text
pld.system.wait(signal, offsets, expected, *, cmp: int) -> Unknown
```

Blocks until this rank's own signal slot of `signal` satisfies the `cmp`
predicate against `expected` (see `WaitCmp`).

Verifier: `signal` must be `DistributedTensorType`; `expected` must be
`ScalarType`; `offsets` must be a `MakeTuple` of rank equal to the signal rank.

## Shared codegen infrastructure

All five ops lower through PTO codegen helpers in
`src/backend/common/pto_ops_common.cpp` and `src/codegen/pto/pto_codegen.cpp`.
The reusable pieces — shared so each op's lowering carries no bespoke peer
arithmetic — are:

| Helper | Role |
| ------ | ---- |
| `CommRemoteOffset_<dtype>` | per-dtype MLIR helper (emitted once by `PTOCodegen::EmitCommRemoteOffsetHelpers`) that turns `(ctx, peer)` into the byte offset of the peer's window slice |
| `EmitCommRemoteView` | emits `CommRemoteOffset + addptr + make_tensor_view` at the call site, yielding the peer-addressed view (used by `remote_load`, `get`'s `src`, and `put`'s `dst`) |
| `EmitPartitionViewPTO` | wraps a tensor view in a full-slice `partition_view` with given offsets/sizes (used by every op for both local and peer operands) |
| `ResolveDistTensorBinding` | resolves a `DistributedTensor` arg to its codegen binding (type + window var) |
| `AsTensorTypeLike` | kind-trait downcast accepting both `TensorType` and `DistributedTensorType` where a view's element/shape info is read uniformly |

The local-vs-remote split is intentional: a *local* operand (e.g. `get`'s
`dst`, `put`'s `src`, `wait`'s `signal`) reuses the tensor view already created by
`EmitMakeTensorViews` with no peer arithmetic, while a *remote* operand (e.g.
`remote_load`'s `target`, `get`'s `src`, `put`'s `dst`) goes through
`EmitCommRemoteView`.

## Pipeline integration

Window buffers and comm groups are collected by the
[`MaterializeCommDomainScopes`](passes/36-materialize_comm_domain_scopes.md) pass, which populates
`CommDomainScopeStmt wrappers in each host_orch body` and the per-window `WindowBuffer` records the runtime
binds physical buffers to.

## Testing

- **IR / parser**: `tests/ut/ir/parser/test_remote_load.py`,
  `tests/ut/ir/parser/test_remote_store.py`, `test_system_ops.py`,
  `test_get_op.py`, `test_put_op.py`, plus the negative verifier coverage
  in `tests/ut/ir/test_distributed_ops.py`.
- **Codegen**: `tests/ut/codegen/distributed/test_distributed_pto_codegen.py`.
- **End-to-end (ST)**: `tests/st/distributed/test_l3_allreduce.py` (mesh
  allreduce with dynamic rank dim `NR = pl.dynamic("NR")`; **P=2** default,
  **P=4** on any four devices (e.g. `--device=0,1,2,3` or `--device=0-3`)),
  `test_l3_allgather.py`,
  `test_l3_reduce_scatter.py`, `test_l3_broadcast.py`, `test_l3_gemm.py`,
  `test_l3_ep_dispatch_combine.py`, `test_l3_notify_wait.py`, and related L3 STs
  under `tests/st/distributed/`. `test_l3_put.py` and `test_l3_get.py` are
  currently **skipped** pending the N7 host codegen (`add_scalar(ctx)` per
  `DistributedTensor`, `ContinuousTensor.make(..., child_memory=True)`) and N8
  driver glue (`HostBufferStaging` window wiring on the codegen-emitted
  `orch.allocate_domain(...)` block). The embedded programs and golden checks
  are the canonical e2e contracts — drop the skip markers once that host-side
  work lands.
