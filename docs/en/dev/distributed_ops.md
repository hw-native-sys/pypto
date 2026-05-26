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

There are **four ops** and **three ABI enums**:

| Op | Direction | Result | Hardware |
| -- | --------- | ------ | -------- |
| `pld.tile.remote_load` | pull (read peer → local tile) | `TileType` | TLOAD |
| `pld.tensor.put` | push (write local → peer) | `Unknown` (side effect) | TPUT |
| `pld.system.notify` | signal a peer's slot | `Unknown` (side effect) | TNOTIFY |
| `pld.system.wait` | block on own slot | `Unknown` (side effect) | TWAIT |

The three side-effect-only ops produce [`UnknownType`](ir/02-types.md): they
exist for their cross-rank effect, not for an SSA value a consumer reads.

## Namespacing: why `tile.*` vs `tensor.*` vs `system.*`

The namespace encodes the IR level the op lives at, not an arbitrary grouping:

- **`pld.tile.remote_load`** produces a *tile* (on-core SRAM region), so it is a
  sibling of `tile.load` and lives in `pld.tile`.
- **`pld.tensor.put`** reads and writes *tensor* (GM) operands — both `dst` and
  `src` are window-bound `DistributedTensor` views and the VEC staging tile
  that TPUT bounces through is synthesised at codegen, never on the DSL
  surface. It is therefore a sibling of `pld.tensor.alloc_window_buffer` /
  `pld.tensor.window`, **not** of the tile-producing `remote_load`.
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

### `pld.tensor.put` (TPUT)

```text
pld.tensor.put(dst, peer, src, *, atomic: int) -> Unknown
pld.tensor.put(dst, peer, src, dst_offsets, src_offsets, shape, *, atomic: int) -> Unknown
```

Synchronously writes the local window-bound `src` into the `peer` rank's slice
of the window-bound `dst`. Both operands are GM-level `DistributedTensor`
views; the VEC staging tile is synthesised at codegen
(`MakePutCodegenPTO` in `src/backend/common/pto_ops_common.cpp`) and never
appears on the DSL surface.

Verifier: `dst` / `src` must both be `DistributedTensorType`; `peer` must be a
`ScalarType`; `dst` and `src` must share element type. The 3-argument form
writes the full local `src` slice to the full peer `dst` slice and therefore
requires identical **positive static** shapes. The 6-argument form writes
`src[src_offsets:src_offsets+shape]` into the peer rank's
`dst[dst_offsets:dst_offsets+shape]`; offsets must match tensor rank and
`shape` must be positive static so the staging VEC buffer has compile-time
extents. `atomic` selects overwrite vs atomic-add (see `AtomicType`).

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

All four ops lower through PTO codegen helpers in
`src/backend/common/pto_ops_common.cpp` and `src/codegen/pto/pto_codegen.cpp`.
The reusable pieces — shared so each op's lowering carries no bespoke peer
arithmetic — are:

| Helper | Role |
| ------ | ---- |
| `CommRemoteOffset_<dtype>` | per-dtype MLIR helper (emitted once by `PTOCodegen::EmitCommRemoteOffsetHelpers`) that turns `(ctx, peer)` into the byte offset of the peer's window slice |
| `EmitCommRemoteView` | emits `CommRemoteOffset + addptr + make_tensor_view` at the call site, yielding the peer-addressed view (used by `remote_load` and `put`'s `dst`) |
| `EmitPartitionViewPTO` | wraps a tensor view in a full-slice `partition_view` with given offsets/sizes (used by every op for both local and peer operands) |
| `ResolveDistTensorBinding` | resolves a `DistributedTensor` arg to its codegen binding (type + window var) |
| `AsTensorTypeLike` | kind-trait downcast accepting both `TensorType` and `DistributedTensorType` where a view's element/shape info is read uniformly |

The local-vs-remote split is intentional: a *local* operand (e.g. `put`'s
`src`, `wait`'s `signal`) reuses the tensor view already created by
`EmitMakeTensorViews` with no peer arithmetic, while a *remote* operand (e.g.
`remote_load`'s `target`, `put`'s `dst`) goes through `EmitCommRemoteView`.

## Pipeline integration

Window buffers and comm groups are collected by the
[`CollectCommGroups`](passes/34-collect_comm_groups.md) pass, which populates
`Program.comm_groups_` and the per-window `WindowBuffer` records the runtime
binds physical buffers to.

## Testing

- **IR / parser**: `tests/ut/ir/parser/test_remote_load.py`,
  `test_system_ops.py`, `test_put_op.py`, plus the negative verifier coverage
  in `tests/ut/ir/test_distributed_ops.py`.
- **Codegen**: `tests/ut/codegen/distributed/test_distributed_pto_codegen.py`.
- **End-to-end (ST)**: `tests/st/distributed/test_l3_remote_load.py`,
  `test_l3_notify_wait.py`, `test_l3_put.py`. These are currently **skipped**
  pending the N7 host codegen (`add_scalar(ctx)` per `DistributedTensor`,
  `ContinuousTensor.make(..., child_memory=True)`) and N8 driver glue
  (`HostBufferStaging` window wiring on the codegen-emitted
  `orch.allocate_domain(...)` block). The embedded
  programs and golden checks are the canonical e2e contracts — drop the skip
  markers once that host-side work lands.
