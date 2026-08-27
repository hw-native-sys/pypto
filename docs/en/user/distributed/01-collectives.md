# Collectives

This page covers the six built-in collectives and when to use each algorithm.
All collectives are **synchronous** across ranks — every rank must call the same
collective with identically shaped signal tensors, or the program hangs or
silently corrupts data.

> **Note:** the code blocks below are illustrative sketches — each shows
> only the primitive calls relevant to that collective and omits setup
> (`my_rank`/`nranks` derivation, buffer allocation, input staging). They
> are not meant to run as-is. For runnable versions, see [Runnable
> Examples](#runnable-examples) below.

## AllReduce

Every rank contributes its local data; every rank receives the reduced
result (`op=` selects `Sum`, `Max`, `Min`, or `Prod`).

```python
# Host orchestrator — simplest form (compiler synthesizes signal).
data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)  # mesh mode, in-place

# InCore kernel — explicit signal.
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="mesh")
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")

# Host orchestrator — spread one call across 4 AIV cores per rank.
data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum, core_num=4)
```

### Mesh Mode

- O(N) remote traffic per step — every rank reads every peer
- One global barrier per call (AtomicAdd/Ge on `[NR, core_num]` signal)
- Works with `pl.dynamic("NR")`
- Best for small messages and low latency

### Ring Mode

- 2(P-1) steps: reduce-scatter + allgather
- O(N/P) remote traffic per step — each rank reads one neighbour
- Signal shape: `[2 × (NR − 1), NR]`
- Requires compile-time-known NR — use a factory function pattern: an outer
  Python function takes `nr`/`size`, derives `total_rounds = 2 * (nr - 1)`,
  and defines the program inside its own body, so `[total_rounds, nr]`
  becomes a compile-time constant.
- Both decorator families support this. `@pl.program` / `@pl.function`
  snapshot the defining frame's locals at decoration time. The `@pl.jit`
  family folds closure constants into the source it regenerates, so a factory
  constant referenced in a HOST orchestrator body resolves the same way. See
  `collectives/test_l3_tensor_allreduce_ring_intrinsic.py` in [Runnable
  Examples](#runnable-examples) for the class form.
- Best for large messages (>16 KiB) and high bandwidth

| Aspect | Mesh | Ring |
| ------ | ---- | ---- |
| Remote traffic per step | O(N) — every rank reads every peer | O(N/P) — each rank reads one neighbour |
| Barrier rounds | 1 (global AtomicAdd/Ge) | 2(P-1) — reduce-scatter + allgather phases |
| Signal shape | `[NR, core_num]` | `[2 × (NR − 1), NR]` |
| Best for | Small messages, low latency | Large messages, high bandwidth |

**Rule of thumb:** Use the default `mode="mesh"`. Switch to `mode="ring"` when
your payload exceeds ~16 KiB and you see mesh bandwidth plateau.

The host orchestrator form (`signal` omitted) is syntactic sugar — the compiler
synthesizes a signal of `[world_size(), core_num]` (mesh only).

### Multi-core (`core_num`)

On a host orchestrator, `core_num` spreads one AllReduce call across several
AIV cores **on each rank**. It does not change the task hierarchy: `device=r`
still selects the card; that rank's builtin task now launches a synchronized
grid of `core_num` blocks that split the payload into 256-element tiles
block-cyclically.

```python
data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum, core_num=4)
```

- Defaults to `1` (single block — the previous behaviour).
- Mesh only: `mode="ring"` requires `core_num == 1`.
- Must not exceed the target's AIV core count (48 on 910B, 36 on 950) — the
  launch requires all blocks to be admitted at once, so an over-subscribed
  request is rejected at compile time.
- An explicit `signal` needs one lane per block: `[world_size(), stride]` with
  `stride >= core_num`. A rank-1 `[world_size()]` signal only works for
  `core_num=1`.
- InCore kernels keep `core_num=1` and use an enclosing `pl.spmd(...)` instead.

### Mutation

`target: InOut` — data is both read (as the reduction input) and written (as
the reduced result). All ranks must pass identically shaped `target` tensors.

### Supported ReduceOp

All four — `Sum`, `Max`, `Min`, `Prod` — on the InCore composite and the HOST
builtin mesh path. The HOST builtin ring path (`builtin.tensor.allreduce_ring`)
is narrower: `Sum` only, with a 4-byte `FP32` target (a compile-time check).
`mesh` targets must be `FP16` or `FP32`; the ring path is `FP32`-only. Every
rank must agree on the same `ReduceOp` and `mode`, on top of the
identically-shaped signal tensors required of every collective.

## Barrier

Cross-rank barrier — blocks until all ranks arrive.

```python
# signal: pld.DistributedTensor[[NR, 1], pl.INT32], freshly allocated.
signal = pld.tensor.barrier(signal)
```

Uses a self-clearing credit barrier (`AtomicAdd(+1)` / `Ge(1)` with a reset
epilogue), so one signal buffer is reusable across back-to-back calls.

## Broadcast

Broadcast root rank's data to all ranks.

```python
# Root stages data before the call.
if my_rank == ROOT_RANK:
    data = pl.store(local, [0, 0], data)
data = pld.tensor.broadcast(data, signal, root=ROOT_RANK)
# Every rank now holds root's data in data[0, 0:SIZE].
```

Root must stage data before the call; non-root slots are ignored on input.
After the call, every rank holds root's data.

## AllGather

Push-based all-gather — every rank pushes its local chunk, every rank receives
the full gathered matrix.

```python
# Stage buffer: this rank's [1, SIZE] chunk (push source).
stage_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
stage = pld.window(stage_buf, [1, SIZE], dtype=pl.FP32)
stage = pl.store(local_input, [0, 0], stage)

# Result buffer: gathered [NR, SIZE] (push target).
data_buf = pld.alloc_window_buffer(NR * SIZE * pl.FP32.get_byte())
data = pld.window(data_buf, [NR, SIZE], dtype=pl.FP32)
sig_buf = pld.alloc_window_buffer(NR * pl.INT32.get_byte())
sig = pld.window(sig_buf, [NR], dtype=pl.INT32)

data = pld.tensor.allgather(stage, data, sig)
# data[src, :] now holds rank src's chunk for every src.
```

`local_data` and `target` **must be different** window buffers. The stage buffer
is the per-rank push source; the target buffer receives the gathered `[NR, SIZE]`
result.

## ReduceScatter

Reduce-scatter: every rank stages all NR chunks, receives its own reduced chunk.

```python
# Signal for the barrier (1-D for host builtins).
sig_buf = pld.alloc_window_buffer(NR * pl.INT32.get_byte())
sig = pld.window(sig_buf, [NR], dtype=pl.INT32)

# Stage all NR chunks into data[NR, SIZE].
for j in pl.range(nranks):
    data = pl.store(chunk_j, [j, 0], data)
data = pld.tensor.reduce_scatter(data, sig, op=pld.ReduceOp.Sum)
# data[my_rank, 0:SIZE] holds this rank's reduced chunk.
```

## AllToAll

Personalized all-to-all exchange — every rank sends a distinct chunk to every
peer and receives a distinct chunk from every peer.

```python
# Stage buffer: push source, [NR, SIZE] with per-destination chunks.
stage_buf = pld.alloc_window_buffer(NR * SIZE * pl.FP32.get_byte())
stage = pld.window(stage_buf, [NR, SIZE], dtype=pl.FP32)
for dest in pl.range(nranks):
    stage = pl.store(chunk_for_dest, [dest, 0], stage)

# Result buffer: push target, [NR, SIZE].
data_buf = pld.alloc_window_buffer(NR * SIZE * pl.FP32.get_byte())
data = pld.window(data_buf, [NR, SIZE], dtype=pl.FP32)
sig_buf = pld.alloc_window_buffer(NR * pl.INT32.get_byte())
sig = pld.window(sig_buf, [NR], dtype=pl.INT32)

data = pld.tensor.all_to_all(stage, data, sig)
# data[src, :] holds the chunk received from rank src.
```

`input` and `target` must be **separate** window buffers.

## InCore vs Host-Level Collectives

PyPTO has three ways to run a collective — pick based on where your code
runs and whether you need `mode="ring"`:

| Aspect | InCore Hand-Rolled | InCore Composite | HOST Builtin |
| ------ | ------------------ | ---------------- | ------------ |
| **Where** | `@pl.jit.incore` | `@pl.jit.incore` | `@pl.jit.host` |
| **How** | Manual `notify`/`wait` + `remote_load` loops | `pld.tensor.allreduce(data, sig, ...)` called directly | `pld.tensor.allreduce(data, [sig,] ...)` called directly |
| **Lowering** | You write the primitives | `LowerCompositeOps` | `LowerHostTensorCollectives` |
| **Modes** | Whatever you implement | `mesh` and `ring` | `mesh` and `ring` (ring: `Sum` + `FP32` only) |
| **Signal shape** | Whatever you allocate | `[nranks, 1]` for mesh (rank count may be dynamic); `[2×(NR−1), NR]` for ring (`NR` must be a compile-time constant) | Mesh: rank-1 `[world_size]` or rank-2 `[world_size, 1]` (the compiler-synthesized signal is rank-2). Ring: `[2*(NR−1)+1, NR]` |
| **When** | Learning, custom protocols | Ring with non-`Sum`/non-`FP32`, or already inside an InCore kernel | Day-to-day host-orchestrated collectives |

Prefer HOST builtins for day-to-day host-orchestrated code — they handle
barrier orchestration and chunking automatically. Only `allreduce` can also
omit the signal argument (the compiler synthesizes one outside loops); the
other five collectives (`barrier`, `broadcast`, `allgather`,
`reduce_scatter`, `all_to_all`) always take an explicit, caller-allocated
signal. Both the InCore composite and the HOST builtin lower `mode="ring"`;
reach for the InCore composite when you need ring with a `ReduceOp` other than
`Sum` or a non-`FP32` dtype, since the HOST builtin ring path is `Sum`+`FP32`
only.

## Runnable Examples

Every collective above has a runnable counterpart under
`tests/st/distributed/` (paths below are relative to that directory). The
[tutorials](05-tutorials.md) are the user-facing counterparts that
build each collective by hand before the builtin is revealed:

| Collective | Tutorial step | Hand-rolled first? |
| ---------- | ------------- | ------------------ |
| barrier | [09-barrier](09-barrier.md) | yes (step 04, then reveal) |
| allreduce | [13-allreduce_mesh](13-allreduce_mesh.md) · [14-allreduce_two_phase](14-allreduce_two_phase.md) · [15-allreduce_ring](15-allreduce_ring.md) · [16-allreduce_reveal](16-allreduce_reveal.md) | yes (steps 08–11) |
| broadcast | planned — step 12 | yes |
| allgather | planned — step 13 | yes |
| reduce_scatter | planned — step 14 | yes |
| all_to_all | planned — step 15 | yes |

| Collective | InCore hand-rolled | InCore composite | HOST builtin |
| ---------- | ------------------ | ---------------- | ------------ |
| allreduce | `collectives/test_l3_allreduce.py` | `collectives/test_l3_tensor_allreduce_intrinsic.py` | `test_l3_host_tensor_allreduce.py` |
| allreduce (ring) | `collectives/test_l3_allreduce_ring.py` | `collectives/test_l3_tensor_allreduce_ring_intrinsic.py` | `test_l3_host_tensor_allreduce_ring.py` |
| barrier | — | `collectives/test_l3_tensor_barrier_intrinsic.py` | `test_l3_host_tensor_barrier.py` |
| broadcast | `collectives/test_l3_broadcast.py` | `collectives/test_l3_tensor_broadcast_intrinsic.py` | `test_l3_host_tensor_broadcast.py` |
| allgather | `collectives/test_l3_allgather.py` | `collectives/test_l3_tensor_allgather_intrinsic.py` | `test_l3_host_tensor_allgather.py` |
| reduce_scatter | `collectives/test_l3_reduce_scatter.py` | `collectives/test_l3_tensor_reduce_scatter_intrinsic.py` | `test_l3_host_tensor_reduce_scatter.py` |
| all_to_all | `collectives/test_l3_all_to_all.py` | `collectives/test_l3_tensor_all_to_all_intrinsic.py` | `test_l3_host_tensor_all_to_all.py` |

## See Also

- [00-model](00-model.md) — Quickstart and model vocabulary
- [02-primitives](02-primitives.md) — The substrate beneath the collectives
- [04-debugging](04-debugging.md) — Common failure patterns
