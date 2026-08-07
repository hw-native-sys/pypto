# Distributed Operators (N6)

## Overview

The N6 distributed op family gives the Python DSL direct, typed access to the
hardware's cross-rank communication primitives. Every op operates against a
**window-bound** [`DistributedTensorType`](ir/02-types.md) — a tensor whose
storage is a slice of a symmetric, per-rank communication window allocated by
`pld.alloc_window_buffer`. Verifiers in this family generally reject plain
`TensorType` (strict kind-trait matching — `As<DistributedTensorType>` does
not match a plain `TensorType`), so a non-window-bound tensor can never be fed
into a cross-rank slot by accident. **Two documented exceptions:**
`pld.tensor.put` (and its lowered `pld.tile.put`) accepts a plain `Tensor` on
the `src` side via `AsTensorTypeLike` — TPUT only needs a readable local GM
region for the source, so kernels can push directly from host-backed inputs
without first staging through a window buffer; `dst` still requires a
window-bound `DistributedTensor`.
`pld.tensor.get` (and its lowered `pld.tile.get`) accepts a plain `Tensor` on
the `dst` side via `AsTensorTypeLike` — TGET only needs a writable local GM
region to receive into, so kernels can TGET directly into host-backed output
tensors; `src` still requires a window-bound `DistributedTensor`.

There are **thirteen ops** and **four ABI enums**:

| Op | Direction | Result | Hardware |
| -- | --------- | ------ | -------- |
| `pld.tile.remote_load` | pull (read peer → local tile) | `TileType` | TLOAD |
| `pld.tile.remote_store` | push (write local tile → peer) | `Unknown` (side effect) | TSTORE |
| `pld.tensor.get` | pull (read peer → local GM) | `Unknown` (side effect) | TGET |
| `pld.tensor.put` | push (write local → peer) | `Unknown` (side effect) | TPUT |
| `pld.tensor.allreduce` | collective reduce over window slices | `DistributedTensorType` (same as src) | builtin collective |
| `pld.tensor.barrier` | synchronise visibility of window data across ranks | `DistributedTensorType` (same as src) | builtin collective |
| `pld.tensor.broadcast` | replicate root rank's data to all ranks | `DistributedTensorType` (same as src) | builtin collective |
| `pld.tensor.reduce_scatter` | reduce and scatter chunks across ranks | `DistributedTensorType` (same as src) | builtin collective |
| `pld.tensor.allgather` | gather data from all ranks via window | `DistributedTensorType` (same as src) | builtin collective |
| `pld.tensor.all_to_all` | push-based symmetric personalized exchange — every rank pushes its per-destination chunks to every peer's window via `pld.tensor.put` (TPUT), returns window as result | `DistributedTensorType` (same as src) | composite / HOST builtin |
| `pld.tensor.all_to_all_v` | variable-size all-to-all (MPI_Alltoallv) — pushes the full MAX_RECV-row block per destination into a flat 2D staging window (transfer size is the full per-destination capacity block), and publishes `min(send_counts[dest], MAX_RECV)` into peer `recv_counts[my_rank, 0]` via `pld.system.notify` (Set) so the receiver can skip rows beyond its count; returns window as result (same window-as-result pattern as symmetric `all_to_all`) | `DistributedTensorType` (same as target) | composite / HOST builtin |
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
- **`pld.tensor.get`** reads and writes *tensor* (GM) operands — `dst` may be a
  window-bound `DistributedTensor` or a plain `Tensor` (TGET only needs a
  writable local GM region to receive into) while `src` must be a window-bound
  `DistributedTensor` (the peer needs a window slot to read from). The VEC
  staging tile that TGET bounces through is materialised by
  `ConvertTensorToTileOps` as an internal `pld.tile.get`, never on the DSL
  surface. It is therefore a sibling of `pld.tensor.alloc_window_buffer` /
  `pld.tensor.window`, **not** of the tile-producing `remote_load`.
- **`pld.tensor.put`** reads and writes *tensor* (GM) operands — `dst` is a
  window-bound `DistributedTensor` (the peer needs a window slot to receive
  into) while `src` accepts either a window-bound `DistributedTensor` or a
  plain `Tensor` (TPUT only needs a readable local GM region on the source
  side). The VEC staging tile that TPUT bounces through is materialised by
  `ConvertTensorToTileOps` as an internal `pld.tile.put`, never on the DSL
  surface. It is therefore a sibling of `pld.tensor.alloc_window_buffer` /
  `pld.tensor.window`, **not** of the tile-producing `remote_load`.
- **`pld.system.notify` / `pld.system.wait`** drive the per-rank signal slot —
  pure control-plane synchronisation with no data operand — so they live in
  `pld.system`.

## ABI enums (`include/pypto/ir/comm.h`)

The four enums are an **append-only ABI**. Their underlying `int` values are
serialised as the op's kwarg payload (`op` for notify, `cmp` for wait, `atomic`
for put) and cast back to the enum at codegen time. New variants may only be
added **at the end** so existing IR and cached programs keep their meaning.

```cpp
enum class NotifyOp : int { kAtomicAdd = 0, kSet = 1 };   // pld.system.notify
enum class WaitCmp  : int { kEq = 0,        kGe = 1 };     // pld.system.wait
enum class AtomicType : int { kNone = 0,    kAdd = 1 };    // pld.tensor.put
enum class ReduceOp : int { kSum = 0, kMax = 1, kMin = 2, kProd = 3 };  // pld.tensor.allreduce
```

| Enum | Variant | Meaning |
| ---- | ------- | ------- |
| `NotifyOp` | `kAtomicAdd` | atomically add `value` into the peer's signal slot |
| `NotifyOp` | `kSet` | non-atomic store of `value` into the peer's signal slot |
| `WaitCmp` | `kEq` | block until `*signal_slot == expected` |
| `WaitCmp` | `kGe` | block until `*signal_slot >= expected` |
| `AtomicType` | `kNone` | plain remote store — overwrite the peer's dst slice |
| `AtomicType` | `kAdd` | atomically add the source data into the peer's dst slice |
| `ReduceOp` | `kSum` | sum-reduce every participating rank's window slice |
| `ReduceOp` | `kMax` | max-reduce every participating rank's window slice |
| `ReduceOp` | `kMin` | min-reduce every participating rank's window slice |
| `ReduceOp` | `kProd` | product-reduce every participating rank's window slice |

Each enum is mirrored across three layers (C++ `enum class` → `nb::enum_` in the
bindings → `.pyi` stub) and surfaced to the DSL as `pld.NotifyOp` /
`pld.WaitCmp` / `pld.AtomicType` / `pld.ReduceOp`. The deducer validates the packed `int`
against the enum range so codegen can cast back without a second guard.

## Barrier-signal protocol

Every `pld.tensor.*` collective (`allreduce`, `barrier`, `broadcast`,
`reduce_scatter`, `allgather`, `all_to_all`, `all_to_all_v`) synchronises through one shared,
**self-clearing credit barrier** built from `pld.system.notify` /
`pld.system.wait`:

```text
Body:      barrier(1); barrier(2); ...; barrier(N)   # g counted within this
                                                      # call only
  barrier(g):
    for peer != my_rank: notify(signal, peer, <my cell>, 1, op=AtomicAdd)
    for src  != my_rank: wait  (signal, <src cell>, g,   cmp=Ge)

Epilogue:  for src != my_rank:
               notify(signal, my_rank, <src cell>, -N, op=AtomicAdd)
```

`AtomicAdd` turns each cell into a credit counter: every notify is a
producer's `+1`, and the epilogue is the sole consumer's `-N`. Because adds
and subtracts are atomic and commutative, the signal is provably all-zero
again once every rank has finished its epilogue for a call — **the signal
carries no state that outlives one call**, so every call's generation `g`
restarts at 1 and no cross-call bookkeeping is required. A slow rank can
inflate a fast rank's own next-call credit by at most 1 while it finishes the
current call (bounded skew), so the counter never overflows and a fast rank
can never observe a spurious pass.

`Ge` (not `Eq`) is load-bearing: a fast peer can advance a cell past the value
the waiting rank is looking for before that rank ever polls it, so an
equality wait would never unblock. For the same reason `Set` must never be
mixed with `AtomicAdd` on the same cells — a set could clobber an already
advanced counter.

`N` (the credit total the epilogue subtracts) may be a **runtime scalar** —
`pld.system.notify`'s `value` only requires `ScalarType` — so a mesh
allreduce's per-chunk credit count does not need to be known at compile time.

**Constraints:**

| Constraint | Why |
| ---------- | --- |
| One signal must not be shared between mesh (`[NR, 1]`) and ring (`[2*(NR-1), NR]`) allreduce | Mesh addresses `[rank, 0]`; ring addresses `[row, rank]` — a shape mismatch, checked at lowering time |
| A call aborted mid-flight (error / timeout) leaves the signal non-zero | Credits leak; recover via a host-side reset (`reset_persistent_windows`) before the next dispatch |

Because the protocol is call-local and the signal always starts a call at
all-zero, collectives are legal inside `for` / `while` / `if` — each call is a
closed cycle, so the same compile-time `expected` values are reused every
iteration. The only remaining requirement is rank-uniform execution (inherent
to any barrier): rank-divergent control flow deadlocks, surfaced by `TWAIT`'s
spin-count assert.

## Op reference

### `pld.tile.remote_load` (TLOAD)

```text
pld.tile.remote_load(target, peer, offsets, shape[, valid_shape])
    -> TileType(shape, target.dtype)
```

Reads a region of the `peer` rank's slice of a window-bound `DistributedTensor`
into a local tile. Mirrors `tile.load` at the IR level (positional `offsets` /
`shape` tuples, `TileType` result) but the source is a *remote* slice — the
address translation is realised at codegen by
`CommRemoteOffset(ctx, peer) + addptr + make_tensor_view`.

`valid_shape` is optional. With or without it, type inference intersects the
requested window with the source tensor's effective valid region and checks
provable physical bounds. When present, `shape` remains the physical UB tile
allocation while `valid_shape` additionally limits the remote partition and
the tile's valid extent. This is the fixed-width ragged-tail form used by
chunked collectives.

Every symbolic source or requested valid extent that survives inference must
be runtime-bound by a kernel scalar, loop variable, or physical tensor-shape
parameter. A symbol that appears only in type metadata is rejected during PTO
codegen.

Verifier: `target` must be `DistributedTensorType`; `peer` must be a
`ScalarType` rank index; `offsets` / `shape` / optional `valid_shape` must each
be a `MakeTuple` whose rank equals `target.shape.size()`.

DSL (`python/pypto/language/distributed/op/tile_ops.py`) accepts positional or
keyword arguments; the IR op keeps them positional, matching `tile.load`.

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
pld.tensor.put(dst, peer, src, *, atomic: int,
               chunk_rows: int = 0, chunk_cols: int = 0, pipeline: bool = False) -> Unknown
pld.tensor.put(dst, peer, src, dst_offsets, src_offsets, shape,
               *, atomic: int, chunk_rows: int = 0, chunk_cols: int = 0, pipeline: bool = False) -> Unknown
```

Synchronously writes local `src` data into the `peer` rank's slice of the
window-bound `dst`. `dst` is a GM-level `DistributedTensor` view; `src` may be
either a `DistributedTensor` view *or* a plain `Tensor` — TPUT only requires a
readable local GM region on the source side, so kernels can push directly from
host-backed inputs without first staging through a window buffer. The VEC
staging tile is materialised by `ConvertTensorToTileOps` as an internal
`tile.create + pld.tile.put`, so it flows through PyPTO's memory allocator and
never appears on the DSL surface.

With no offsets/shape this writes the full local `src` slice to the full peer
`dst` slice. Supplying `dst_offsets`, `src_offsets`, and `shape` narrows the
transfer to matching subregions; all three must be provided together.

**Staging-tile chunking.** By default the staging tile spans the whole
flattened transfer `[rows, cols]` extent (`rows` = product of the leading dims,
`cols` = the innermost dim), so a transfer must fit in UB. The optional
`chunk_rows` / `chunk_cols` attrs (`0` = full) shrink the staging tile to a
sub-tile of that extent; the codegen keeps the `pto.comm.tput` partition views
at the **full** transfer extent and pto-isa TPUT 2-D-slides the transfer through
the smaller stage. This lets a single `put` move data larger than UB without the
caller writing an explicit chunk loop. Oversized chunk values are clamped to the
transfer extent.

**Double-buffering (`pipeline`).** Setting `pipeline=True`
makes `ConvertTensorToTileOps` materialise **two** identical VEC staging tiles
(`tput_stage_ping` / `tput_stage_pong`) and thread both into `pld.tile.put` as a
second `stage` operand. The codegen then emits the ping-pong form
`pto.comm.tput(dst_pv, src_pv, buf(%ping, %pong) : …)`, which PTOAS routes to
pto-isa's double-buffered `TPUT` overload — it overlaps the TLOAD of the next
chunk with the TSTORE of the previous one across the two tiles. Because the
benefit only exists when the transfer is chunked into more than one piece,
`pipeline` **requires both `chunk_rows` and `chunk_cols` to be set** (the deducer
and the DSL both reject `pipeline` without a full chunk). The two tiles are
distinct `tile.create` allocations, so the memory allocator gives them
non-overlapping UB addresses (pto-isa's ping/pong requirement).

**Dynamic transfer extent.** The transfer may be **dynamic** — either the
subregion `shape` (a runtime sub-extent of the window) or the `dst` / `src`
window (`DistributedTensorType`) dims themselves, for a full-slice transfer.
pto-isa reads the extent from the partition views at runtime, so the codegen
emits a dynamic partition view (`<?x…>`) and chunks it. A dynamic flattened
transfer dim must be bounded by the corresponding static chunk, because the VEC
staging tile is statically allocated: a dynamic innermost dim requires
`chunk_cols`, a dynamic leading dim requires `chunk_rows`. For a full-slice
transfer the `dst` and `src` dims must match — by value when static, structurally
when dynamic.

Verifier: `dst` must be `DistributedTensorType`; `src` must be either
`TensorType` or `DistributedTensorType` (matched via `AsTensorTypeLike`);
`peer` must be a `ScalarType`; `dst` and `src` must share element type, rank,
and **positive** dimensions (positivity checked on static dims; dynamic dims are
allowed and bounded by the chunk). Full-slice `put` requires matching `dst` /
`src` shape; subregion `put` allows different full slice extents as long as the
explicit transfer region is in bounds (checked on static dims). Any dynamic
transfer dim requires a matching static chunk (see above). `atomic` selects
overwrite vs atomic-add (see `AtomicType`). The lowered `pld.tile.put` verifier
requires the staging tile to **fit within** the flattened transfer in both
**static** dims (it may be smaller — a chunk — but never larger; dynamic dims
are bounded by the chunk at runtime).

### `pld.tensor.get` (TGET)

```text
pld.tensor.get(dst, peer, src, *, chunk_rows: int = 0, chunk_cols: int = 0, pipeline: bool = False) -> Unknown
pld.tensor.get(dst, peer, src, dst_offsets, src_offsets, shape,
               *, chunk_rows: int = 0, chunk_cols: int = 0, pipeline: bool = False) -> Unknown
```

Synchronously reads the `peer` rank's slice of the window-bound `src` into the
local `dst`. `dst` may be a window-bound `DistributedTensor` or a plain
`Tensor`; `src` must be a window-bound `DistributedTensor`. The VEC staging
tile is materialised by `ConvertTensorToTileOps` as an internal
`tile.create + pld.tile.get`, so it flows through PyPTO's memory allocator and
never appears on the DSL surface.

With no offsets/shape this reads the full peer `src` slice into the full local
`dst` slice. Supplying `dst_offsets`, `src_offsets`, and `shape` narrows the
transfer to matching subregions; all three must be provided together. The
optional `chunk_rows` / `chunk_cols` attrs (`0` = full) shrink the staging tile
to a sub-tile of the flattened transfer extent so pto-isa TGET auto-chunks the
full transfer through it — same contract as `put` above, including a **dynamic
transfer** (the subregion `shape` or the full-slice `dst` / `src` window dims)
bounded by a matching static chunk (dynamic innermost needs `chunk_cols`,
dynamic leading needs `chunk_rows`). Setting `pipeline=True`
double-buffers the chunked read through two staging tiles
(`tget_stage_ping` / `tget_stage_pong`), emitting
`pto.comm.tget(…, buf(%ping, %pong) : …)` for pto-isa's ping-pong `TGET`
overload — same contract as `put`, and likewise **requires both `chunk_rows` and
`chunk_cols`**.

Verifier: `dst` must be either `TensorType` or `DistributedTensorType` (matched
via `AsTensorTypeLike`); `src` must be `DistributedTensorType`; `peer` must be a
`ScalarType`; `dst` and `src` must share element type, rank, and **positive**
dimensions (positivity checked on static dims; dynamic dims allowed, bounded by
the chunk). Full-slice `get` requires matching `dst` / `src` shape; subregion
`get` allows different full slice extents as long as the explicit transfer
region is in bounds (checked on static dims); any dynamic transfer dim requires
a matching static chunk. Besides `chunk_rows` / `chunk_cols`, `get` accepts no
keyword attributes.

### `pld.tensor.all_to_all_v`

```text
pld.tensor.all_to_all_v(
    input, target, signal, send_counts, recv_counts
) -> DistributedTensorType(target)
```

Variable-size all-to-all (MPI_Alltoallv). Flat 2D layouts:

- `input` — Tensor or DistributedTensor `[NR*MAX_RECV, SIZE]`
- `target` — DistributedTensor `[NR*MAX_RECV, SIZE]` (window-as-result)
- `signal` — DistributedTensor INT32 `[NR, 1]` (single-use Set(1)/wait≥1 barrier)
- `send_counts` — Tensor-like INT32 `[NR]` or `[NR, 1]` (runtime rows per dest)
- `recv_counts` — DistributedTensor INT32 `[NR, 1]` (InOut recvcounts)

`MAX_RECV = target.shape[0] // NR`. Lowering reads `send_counts[dest]` at
runtime, clamps to `MAX_RECV`, and publishes the **clamped** count into peer
`recv_counts[my_rank, 0]` via `pld.system.notify` (Set). The push itself
always transfers the full `MAX_RECV`-row capacity block per destination —
independent of the runtime count — so rows beyond a sender's actual count
still cross the wire; after the barrier the receiver uses `recv_counts[src, 0]`
to skip those rows (MPI_Alltoallv semantics apply to the logical result, not
the wire transfer). On the InCore path the transfer is a compile-time-sized
`pld.tile.put` (PTOAS requires static partition-view dims); on the HOST path
the kernel derives `MAX_RECV` at entry from the runtime rank count
(`target.shape[0] / nranks`), so it is always consistent with the devices
actually running.

**InCore composite** (`LowerCompositeOps`): the primitive above, decomposed
into `pld.tile.put` + `pld.system.notify`/`wait` inside a chip kernel.

**HOST builtin** (`LowerHostTensorCollectives`): the same 5-arg call, made
from a `host_orch` function, lowers per-device to `builtin.tensor.all_to_all_v`
— an in-kernel-TPUT AIV builtin following the same pattern as
`builtin.tensor.all_to_all`. `input` and `send_counts` must both be
window-bound `DistributedTensor`s at this layer (narrower than the composite's
`AsTensorTypeLike`, forced by the HOST dispatch codegen, which only supports
window-bound or tile args) — all five operands (`input`, `target`, `signal`,
`send_counts`, `recv_counts`) must resolve to pairwise-distinct window
allocations (aliasing any pair is a cross-process race: data-vs-data is a TPUT
overwrite, data-vs-control is a notify/count write racing a kernel read,
control-vs-control is a notify racing a count publish). The kernel derives
`MAX_RECV` at entry as `target.shape[0] / nranks` (the runtime comm-domain
size), so the block layout is always consistent with the devices actually
running — no exact `signal.shape[0]` == device-count requirement and no
per-`MAX_RECV` variant mangling. Not supported inside a `for`/`while` loop in
`host_orch` (single-use signal protocol) — the same restriction
`LowerCompositeOps` enforces on the InCore path.

### `pld.tensor.allreduce`

```text
pld.tensor.allreduce(src, *, op: ReduceOp = ReduceOp.Sum, mode: str = "mesh", core_num: int = 1) -> DistributedTensorType(src)
pld.tensor.allreduce(src, signal, *, op: ReduceOp = ReduceOp.Sum, mode: str = "mesh", core_num: int = 1) -> DistributedTensorType(src)
```

Reduces every participating rank's window-bound `src` slice in place and returns
the same type as `src`. The `mode` keyword selects the lowering algorithm:

Fully-valid packed mesh targets are viewed as one logical `[1, N]` stream and
processed in UB chunks of at most 16 KiB. A statically known `N` smaller than
that budget shrinks the physical chunk to the smallest 32-byte-aligned width
that covers `N`; larger or dynamic inputs retain the maximum width. The last
chunk keeps its selected physical width
but carries `valid_shape=[1, min(chunk, N-offset)]`, so arbitrary element counts
neither read nor store past the end.

For mesh lowering, a partial `TensorView.valid_shape` is preserved for packed ND
targets when its valid box can be represented by collapsing the leading dimensions
to one 2D rectangle and a statically bounded physical tile fits within one 16-KiB
chunk. A symbolic valid extent falls back to the source's physical rectangle when
that rectangle fits the budget. Oversized partial rectangles,
strided targets, DN partial views, and non-representable partial boxes are rejected
explicitly.

Any symbolic target or partial-valid extent that survives lowering must be
runtime-bound by a kernel scalar, loop variable, or physical tensor-shape
parameter; a type-metadata-only symbol is rejected during PTO codegen. A fully
dynamic physical target dimension is bound from that tensor parameter.

- **`"mesh"` (default)** — direct all-to-all exchange with O(P) HCCL windows.
  Signal shape `[NR, 1]` (one cell per rank). A ready barrier (generation 1)
  precedes the chunk loop. Each chunk performs `remote_load+accumulate`, then
  a barrier on its own call-local generation before store-back, preventing
  write-after-read races. A self-clearing epilogue then subtracts the call's
  total credit count back out of every cell (see
  [Barrier-signal protocol](#barrier-signal-protocol)), so the signal is
  all-zero again once the call completes.
- **`"ring"`** — NCCL-style chunked reduce-scatter + allgather schedule with
  O(1) HCCL windows.  Signal shape `[2 * (NR − 1), NR]` (one row per ring
  round, one cell per rank). A packed ND target is viewed as one logical
  `[1, SIZE]` stream; a partial valid box must be a contiguous row-major
  prefix. Lowering keeps the full physical
  `[1, product(target.shape)]` view and records the logical prefix as
  `TensorView.valid_shape=[1, product(target.valid_shape)]`. FP32 divides
  `SIZE` with balanced `floor(i * SIZE / NR)` boundaries. FP16 rounds each
  interior boundary up to 16 elements (32 bytes) and caps it at `SIZE`, so
  every non-empty segment begins at an MTE-safe address without changing the
  packed user-visible layout. Empty segments remain legal for very short
  inputs. Each segment is processed in at most 16-KiB physical subchunks; an
  FP16 ragged remote tail rounds only its physical read span to 32 bytes and
  restores the logical `valid_shape` before reduction or store. Every
  subchunk uses ready and read-complete barriers on its own call-local
  generation before store-back, preventing write-after-read races. A
  self-clearing epilogue subtracts the call's total credit count back out of
  every row of the signal afterward.

Host-orchestrator user code may omit `signal` outside `for` and `while` loops;
the [`SynthesizeAllReduceSignals`](passes/41-synthesize_allreduce_signals.md)
pass inserts a private INT32 signal window with semantic shape
`[world_size, core_num]`
for that call (mesh mode only — `mode="ring"` requires an explicit signal). The
pass binds `world_size = pld.world_size()` as a standalone statement and uses
that variable in the synthesized buffer size and window shape. The
self-clearing protocol (see [Barrier-signal protocol](#barrier-signal-protocol))
makes every call a stateless cycle, so calls inside `for` / `while` loops are
supported like any other collective. Explicit
`signal` remains the internal form used by InCore lowering and by tests that
intentionally construct the internal protocol. Comm-domain materialisation then
keeps the signal buffer in the same domain as `src`, even when it is not passed
to a user chip kernel. Mesh, ring, and host-builtin paths support FP16 and FP32
with `ReduceOp.Sum`, `Max`, `Min`, and `Prod` for arbitrary positive element
counts. InCore lowering uses UB-bounded chunks; the host builtin uses
256-element chunks. InCore mesh and ring round only the physical FP16 remote
tail span to 32 bytes. The host builtin rounds ragged FP16 and FP32 load spans
to 32 bytes. Both preserve the logical valid shape. The host builtin accepts
either a rank-1 `[world_size]` signal or a rank-2
`[world_size, signal_stride]` signal. Ring mode (`mode="ring"`) for the host
orchestrator lowers to `builtin.tensor.allreduce_ring` and requires an explicit
rank-2 `[2 * (NR - 1) + 1, NR]` INT32 signal (one extra row for the return
barrier).

#### Host multi-core AllReduce (`core_num`)

`core_num` selects how many AIV blocks one HOST `pld.tensor.allreduce` dispatch
uses **on each rank**. It does not change the task hierarchy: `device=r` still
selects the card and the call still lowers to one builtin orchestration task per
rank; that task now launches a synchronized SPMD grid of `core_num` blocks.

```python
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, core_num=4)
```

| Constraint | Rule |
| ---------- | ---- |
| Range | Positive compile-time integer, default `1` (pre-existing behavior) |
| Schedule | Mesh only — `mode="ring"` requires `core_num == 1` |
| Capacity | At most the backend's AIV core count (submitted via `rt_submit_aiv_task`, so one block = one AIV core) |
| InCore | Must stay `1`; use an enclosing `pl.spmd(...)` for multi-core work |

**Signal layout.** The signal is a peer-major, lane-contiguous
`[world_size, signal_stride]` matrix with `signal_stride >= core_num`. Block `b`
waits on `signal_base + peer * signal_stride + b` and notifies peer `p` at
`signal_base + my_rank * signal_stride + b`, so every `(peer, block)` pair owns
an independent counter. A rank-1 `[world_size]` signal (stride 1) is valid only
for `core_num == 1`. A synthesized signal is exactly `[world_size, core_num]`;
an explicit signal may be wider.

**Kernel partitioning.** Blocks own 256-element tiles block-cyclically — block
`b` processes tiles `b, b + C, b + 2C, ...` for `C` launched blocks — so no two
blocks touch the same chunk. Each block runs the ready barrier once, then a
read-done barrier per chunk. That per-chunk barrier must stay **before** the
store: without it a rank could overwrite its source chunk before the matching
block on another rank has remote-loaded it. Blocks with no data still run the
ready barrier, keeping ranks symmetric and letting `core_num` exceed the chunk
count. Blocks at or beyond `signal_stride` have no lane to own, so they retire
immediately without joining the barrier; `signal_stride` is equal on every rank,
so all ranks retire the same blocks and the protocol stays symmetric.

**Why one SPMD grid rather than `pl.parallel`.** `pl.parallel(N)` emits `N`
independent tasks, each with its own TaskId and scheduling lifetime — unsafe for
an in-place collective. Ranks may schedule chunk tasks in different orders, so a
task waiting on another rank's matching chunk can deadlock, and conservative
dependency tracking on the shared InOut window tends to serialize them anyway.
One SPMD grid avoids both: `require_sync_start` admits all blocks together and
`block_idx` gives deterministic, matching partitioning on every rank. That is a
per-card admission guarantee, not a global simultaneous start across ranks — the
ready barrier absorbs cross-rank launch skew.

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
`src/backend/common/pto_ops_distributed.cpp` and `src/codegen/pto/pto_codegen.cpp`.
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

Comm domains and their slot allocations are materialised by the
[`MaterializeCommDomainScopes`](passes/42-materialize_comm_domain_scopes.md) pass, which wraps each
host_orch body in nested `CommDomainScopeStmt` nodes (one per inferred comm domain) and produces the
per-window `WindowBuffer` records that the runtime binds physical buffers to.
Host-level tensor collectives are then lowered by
[`LowerHostTensorCollectives`](passes/43-lower_host_tensor_collectives.md) into internal builtin chip
dispatches before the final `Simplify`.

## Testing

- **IR / parser**: `tests/ut/ir/parser/test_remote_load.py`,
  `tests/ut/ir/parser/test_remote_store.py`, `test_system_ops.py`,
  `test_get_op.py`, `test_put_op.py`, plus the negative verifier coverage
  in `tests/ut/ir/test_distributed_ops.py`.
- **Codegen**: `tests/ut/codegen/distributed/test_distributed_pto_codegen.py`.
- **End-to-end (ST)**: `tests/st/distributed/test_l3_allreduce.py` (mesh
  allreduce with dynamic rank dim `NR = pl.dynamic("NR")`; **P=2** default,
  **P=4** on any four devices (e.g. `--device=0,1,2,3` or `--device=0-3`)),
  `test_l3_allgather.py`, `test_l3_reduce_scatter.py`, `test_l3_broadcast.py`
  (each likewise dynamic-NR, P=2/P=4),
  `test_l3_tensor_allreduce_intrinsic.py`, `test_l3_tensor_allreduce_ring_intrinsic.py`,
  `test_l3_allreduce_ring.py` (hand-rolled ring RS+AG), `test_l3_host_tensor_allreduce.py`,
  `test_l3_host_tensor_allreduce_ring.py`,
  `test_l3_ep_dispatch_combine.py`, `test_l3_notify_wait.py`,
  `test_l3_tensor_all_to_all_v_intrinsic.py` (InCore composite),
  `test_l3_host_tensor_all_to_all_v.py` (HOST builtin), and related L3 STs
  under `tests/st/distributed/`. **Put/get canonical e2e contracts** are now
  enabled: `test_l3_put.py` (ring overwrite, row-offset put, atomic-add put, and
  chunked/pipelined transfers ✅), `test_l3_get.py` (ring read, row-offset get ✅),
  and `test_l3_remote_store.py` (tile-level subview push ✅). All tests use the
  `pld.system.notify` / `pld.system.wait` handshake pattern established by
  notify/wait and collective STs.
