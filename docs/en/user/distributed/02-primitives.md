# Primitives

Most users call `pld.tensor.*` collectives directly — reach for these
lower-level primitives only when building a custom protocol.

> **Note:** the notify/wait, put/get, and remote-load/store code blocks
> below are illustrative sketches — they omit `nranks`/`my_rank`
> derivation and buffer setup and are not meant to run as-is. For
> runnable versions, see [Runnable Examples](#runnable-examples) below.

## Types and Enums

| Name | Values | Description |
| ---- | ------ | ----------- |
| `NotifyOp` | `AtomicAdd`, `Set` | Signal deposit mode. `AtomicAdd`: atomically increment the peer's signal slot (use for multi-rank barriers). `Set`: overwrite the peer's signal slot (use for 1:1 handshakes). |
| `WaitCmp` | `Eq`, `Ge` | Wait predicate. `Eq`: block until signal slot equals expected value. `Ge`: block until signal slot >= expected value. |
| `ReduceOp` | `Sum`, `Max`, `Min`, `Prod` | Reduction operator for collective operations. Support is per-operation: `allreduce` accepts all four; `reduce_scatter` accepts only `Sum` and rejects the rest at the deducer. |
| `AtomicType` | `None_`, `Add` | Remote-store combine mode. `None_`: plain store. `Add`: atomically accumulate into peer's destination. |
| `DistributedTensor` | — | A tensor view bound to a comm-domain window buffer. Every collective and RMA op requires this type on the window side. |
| `CommCtx` | — | Communication context handle. Produced by `get_comm_ctx()`; consumed by `rank()` and `nranks()`. |

## System Substrate (`pld.system.*`)

These are the lowest-level distributed primitives. Scope varies per op —
`world_size` is host-only; `get_comm_ctx` works in both host orchestrator and
InCore kernel code; `rank`, `nranks`, `notify`, and `wait` have codegen
support only in InCore kernel code (there is no host-orchestrator lowering
for them).

| Name | Signature | Description |
| ---- | --------- | ----------- |
| `world_size` | `() -> Scalar` | **Host-only.** Number of ranks in the distributed execution. Returns `INT64`. |
| `get_comm_ctx` | `(dist_tensor: DT) -> Ctx` | Lift a `DistributedTensor` to its `CommCtx` handle. Works in host orchestrator and InCore code. The verifier rejects plain `pl.Tensor`. |
| `rank` | `(ctx: Ctx) -> Scalar` | **InCore-only.** Local rank index (`INT32`). Lowers to a load of `CommContext::rankId`. |
| `nranks` | `(ctx: Ctx) -> Scalar` | **InCore-only.** Number of ranks in this comm group (`INT32`). Lowers to a load of `CommContext::rankNum`. |
| `notify` | `(target: DT, peer: IntLike, offsets: Sequence[IntLike], value: IntLike, *, op: NotifyOp) -> Call` | **InCore-only.** Cross-rank signal deposit. **Side-effect-only** — no return value. Lowers to `TNOTIFY`. |
| `wait` | `(signal: DT, offsets: Sequence[IntLike], expected: IntLike, *, cmp: WaitCmp) -> Call` | **InCore-only.** Cross-rank wait. **Side-effect-only** — blocks until the local signal slot satisfies `cmp(expected)`. Lowers to `TWAIT`. |

## Window Buffer Management (`pld.tensor.*`)

`window` and `alloc_window_buffer` live in `pld.tensor.*`, not `pld.system.*`,
even though they are as foundational as the substrate above.

| Name | Signature | Description |
| ---- | --------- | ----------- |
| `window` | `(buf: Ptr, shape: Sequence[IntLike], *, dtype: DataType) -> DT` | Materialise a window-buffer `Ptr` as a `DistributedTensor` view. `buf` comes from `alloc_window_buffer`. |
| `alloc_window_buffer` | `(size: IntLike, *, name: str = "") -> Ptr` | Allocate a per-rank HCCL window buffer. **Size is in bytes.** The `name` kwarg is injected by the parser from the LHS assignment — never pass it explicitly. |
| `alloc_window_buffer` | `(shape: Sequence[IntLike], *, dtype: DataType, name: str = "") -> Ptr` | Convenience overload. `size = prod(shape) x dtype.get_byte()` computed automatically. |

## Notify & Wait: The Signal Handshake

The lowest-level synchronisation primitive. Each rank writes to a peer's
signal cell, then blocks until its own cell has been written.

```python
@pl.jit.incore
def handshake_step(
    out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
    signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
    peer: pl.Scalar[pl.INT32],
    tag: pl.Scalar[pl.INT32],
) -> pl.Tensor[[1, 1], pl.INT32]:
    # 1. Write our tag into the peer's signal cell.
    pld.system.notify(
        signal, peer=peer, offsets=[0, 0],
        value=tag, op=pld.NotifyOp.Set,
    )

    # 2. Wait until our own cell has been written.
    pld.system.wait(
        signal=signal, offsets=[0, 0],
        expected=1, cmp=pld.WaitCmp.Ge,
    )

    # 3. Read the received tag back out.
    received = pl.read(signal, [0, 0])
    pl.write(out, [0, 0], received)
    return out
```

> The `wait` uses `Ge` with `expected=1`, which means the peer's `tag`
> **must be >= 1**. Passing `tag=0` will cause a permanent hang.

### Choosing NotifyOp and WaitCmp

| Scenario | NotifyOp | WaitCmp | Why |
| -------- | -------- | ------- | --- |
| 1:1 exchange (one writer per slot) | `Set` | `Eq` or `Ge` | Atomic increment not needed — overwrite is clear and fast. |
| N-to-1 barrier (many writers, one slot) | `AtomicAdd` | `Ge` | Every writer atomically adds its contribution. The sum increments monotonically; wait for the expected total. |
| Multi-round protocol | `AtomicAdd` | `Ge` | The counter advances across rounds without reset — each round uses a fresh row or the caller re-allocates the buffer. |

**Expected output** for 2 ranks: rank 0 writes tag=2, waits for tag 1 from rank 1:
`outputs[0] == 1`. Rank 1 writes tag=1, waits for tag 2 from rank 0:
`outputs[1] == 2`. Result: `outputs == [[1], [2]]`.

> **Buffer re-use safety:** Signal cells are zero-initialised by
> `alloc_window_buffer`. After `notify`, the signal cell holds the written
> value; after `wait` returns, the caller has observed the barrier. Do not
> reuse the same signal buffer across back-to-back collectives — the protocol
> uses monotonic counters that do not self-reset. Allocate a fresh buffer.

## Tile-Level RMA (`pld.tile.*`)

Low-level cross-rank remote memory access. These are tile-level primitives used to build
collectives; most users call `pld.tensor.*` collectives instead.

| Name | Signature | Description |
| ---- | --------- | ----------- |
| `remote_load` | `(target: DT, peer: IntLike, offsets: Sequence[IntLike], shape: Sequence[IntLike], valid_shape=None) -> Tile` | Load a region of peer rank's `DT` into a local tile. `shape` defines the tile dimensions. `valid_shape` keeps the physical tile fixed-size while a ragged tail reads only real data. Offsets must match what the peer stored — a 1-element misalignment causes silent corruption. |
| `remote_store` | `(src_tile: Tile, target: DT, peer: IntLike, offsets: Sequence[IntLike]) -> Call` | Write a local tile into peer rank's `DT`. Side-effect-only. |

## Put and Get (`pld.tensor.*`)

One-sided bulk transfer — rank A writes to or reads from rank B's window
without rank B participating in the transfer (beyond the signal barrier).

### Put (Write to Peer)

| Name | Signature | Mutation | Description |
| ---- | --------- | -------- | ----------- |
| `put` | `(dst: DT, peer: IntLike, src: DT \| Tensor, dst_offsets=None, src_offsets=None, shape=None, *, atomic=AtomicType.None_, chunk_rows=0, chunk_cols=0, pipeline=False) -> Call` | `dst: InOut`, `src: In` | Write local `src` into peer rank's `dst`. `dst` **must** be window-bound; `src` may be plain `Tensor`. With no offsets/shape, writes the full local slice. `atomic=Add` accumulates instead of overwriting. |

### Get (Read from Peer)

| Name | Signature | Mutation | Description |
| ---- | --------- | -------- | ----------- |
| `get` | `(dst: DT \| Tensor, peer: IntLike, src: DT, dst_offsets=None, src_offsets=None, shape=None, *, chunk_rows=0, chunk_cols=0, pipeline=False) -> Call` | `dst: Out`, `src: In` | Read peer rank's `src` into local `dst`. `src` **must** be window-bound; `dst` may be plain `Tensor`. |

### Chunking and Pipelining Constraints

`chunk_rows`/`chunk_cols` (`0` = full extent) shrink the staging tile so a
transfer larger than the on-chip staging budget still moves in one call,
sliding through the smaller stage automatically.

> **Fatal pitfall:** `pipeline=True` **requires both `chunk_rows > 0` and
> `chunk_cols > 0`** — the double-buffering benefit only exists when the
> transfer is actually chunked. Passing `pipeline=True` with either chunk
> dimension left at `0` raises a `ValueError` before dispatch.

A **dynamic** transfer extent (a runtime-sized `shape`, or a full-slice
transfer where `dst`/`src`'s own dims are dynamic) must be bounded by a
matching static chunk: a dynamic innermost dimension requires `chunk_cols`
to be set, and a dynamic leading dimension requires `chunk_rows` to be set —
the staging tile is allocated statically and can't size itself from a
runtime value.

## Writing Your Own Collective

Every built-in collective is a composition of lower-level primitives. The
mesh allreduce is: stage-in -> barrier -> remote-accumulate -> stage-out.

### The Barrier in Isolation

```python
# signal: pld.DistributedTensor[[NR, 1], pl.INT32]
for peer in pl.range(nranks):
    if peer != my_rank:
        pld.system.notify(
            signal, peer=peer, offsets=[my_rank, 0],
            value=1, op=pld.NotifyOp.AtomicAdd,
        )
for src in pl.range(nranks):
    if src != my_rank:
        pld.system.wait(
            signal, offsets=[src, 0],
            expected=1, cmp=pld.WaitCmp.Ge,
        )
```

With `offsets=[my_rank, 0]`, each rank owns a dedicated row — cell `[r, 0]`
in every peer's window has exactly one writer, rank `r` itself, so `Set`
would work identically here. `AtomicAdd` is shown because this is the same
notify call every barrier in this doc uses; the "many writers, one slot"
case that actually requires `AtomicAdd` is a *shared*-cell barrier (see the
table above) — give every rank a distinct offset like this one only when you
need to distinguish *which* peers have arrived, not just *whether* everyone
has.

### Remote Accumulate

```python
acc = pl.load(data, [0, 0], [1, SIZE])
for peer in pl.range(nranks):
    if peer != my_rank:
        peer_tile = pld.tile.remote_load(
            data, peer=peer, offsets=[0, 0], shape=[1, SIZE]
        )
        acc = pl.add(acc, peer_tile)
```

`remote_load` reads the peer's window slice into a local tile. The offset
and shape must match what the peer stored — a mismatch reads garbage.

## 2-Segment vs 3-Segment Namespace

| Short form (`pld.*`) | Full path |
| -------------------- | --------- |
| `pld.world_size()` | `pld.system.world_size()` |
| `pld.rank(ctx)` | `pld.system.rank(ctx)` |
| `pld.nranks(ctx)` | `pld.system.nranks(ctx)` |
| `pld.get_comm_ctx(dt)` | `pld.system.get_comm_ctx(dt)` |
| `pld.alloc_window_buffer(...)` | `pld.tensor.alloc_window_buffer(...)` |
| `pld.window(...)` | `pld.tensor.window(...)` |
| `pld.remote_load(...)` | `pld.tile.remote_load(...)` |
| `pld.remote_store(...)` | `pld.tile.remote_store(...)` |

**No short form:** `pld.notify(...)`, `pld.wait(...)`, `pld.put(...)`,
`pld.get(...)`, `pld.allreduce(...)`, and all other collective ops — these
require the full 3-segment namespace.

## Runnable Examples

The [tutorials](05-tutorials.md) teach each primitive by hand before
any builtin is revealed (steps 03–06 ship; 07–15 are planned):

| Primitive | Tutorial step |
| --------- | ------------- |
| window buffer | [08-window_buffer](08-window_buffer.md) (step 03) |
| notify / wait | [09-barrier](09-barrier.md) (step 04) |
| remote_load / remote_store | [10-remote_load_store](10-remote_load_store.md) (step 05) |
| put / get | [11-put_get](11-put_get.md) (step 06) |

| Primitive | Test |
| --------- | ---- |
| notify / wait | `test_l3_notify_wait.py` |
| put / get | `test_l3_put.py` / `test_l3_get.py` |
| remote_store | `test_l3_remote_store.py` |

(paths relative to `tests/st/distributed/`)

## See Also

- [01-collectives](01-collectives.md) — The collectives built on these primitives
- [03-execution](03-execution.md) — DistributedWorker lifecycle and environment setup
- [04-debugging](04-debugging.md) — Common failure patterns
