# Distributed Tutorials

The `pld` vocabulary taught step by step: a sixteen-step tutorial series, one
concept per program. Eleven runnable examples ship now — from "hello rank"
through point-to-point moves, the dynamic rank count, and the all-reduce trio
plus its reveal; steps 12–15 (the remaining collectives) and step 16
(composition) are planned.

> **Prerequisites:** the [Distributed Programming](../distributed/index.md)
> chapter — read it once for the vocabulary, then come back here to build the
> same ideas by hand. Hardware: two devices for steps 01–06, any count ≥ 2 for
> step 07 (three or more to see the ring differ from P=2), four for the
> collective comparisons in steps 08–11.

## The idea

The chapter reference tells you what `pld` *is*; this tutorial series shows you
what it *does*. Each shipped step is a small, golden-validated program that
teaches exactly one abstraction, and the steps are ordered so you build each
idea from the primitives before a builtin replaces it:

- Steps 01–02 establish the execution model (rank identity, the three levels).
- Step 03 introduces **window memory** — the substrate everything else touches.
- Step 04 builds a **barrier by hand** from `notify`/`wait` before the builtin
  is revealed.
- Steps 05–06 cover **point-to-point** moves (`remote_load`/`remote_store`,
  `put`/`get`).
- Step 07 makes the rank count **dynamic** (`pl.dynamic("NR")`): the same
  source compiles for any P — the mechanism the P=4 collectives build on.
- Steps 08–11 build **all-reduce three ways** (mesh, two-phase, ring) and then
  reveal `pld.tensor.allreduce`.
- Steps 12–15 cover the remaining collectives; step 16 composes three of them.

> **Reveal discipline:** the walkthrough pages do not introduce a builtin
> (`pld.tensor.barrier`, `pld.tensor.allreduce`, …) before the step that
> reveals it — this index only previews what is coming. By the time a builtin
> appears, you have already written the hand-rolled version and know what it
> lowers to.
>
> **Progression:** every step uses only concepts introduced in earlier steps
> (or in the prerequisite chapter). When a step mentions something taught
> later — like step 04's one-line `remote_load` in the barrier reveal — it is
> a pointer, not required knowledge: you can read the later step to meet the
> idea properly.

## Suggested reading order

Read the steps in order — **01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 →
11 → 12 → 13 → 14 → 15 → 16**. Every page repeats this block. Steps 01–11 ship
now; 12–16 remain planned.

## The 16 steps

| Step | Program | Teaches | Status |
| ---- | ------- | ------- | ------ |
| 01 | `01_hello_rank.py` | Rank identity, `pld.world_size()`, `DistributedConfig`; one per-rank dispatch | ✅ shipped |
| 02 | `02_programming_model.py` | The three levels: `@pl.jit.host` → `@pl.jit` → `@pl.jit.incore` | ✅ shipped |
| 03 | `03_window_buffer.py` | Window memory: `alloc_window_buffer`/`window`; own slice, no communication | ✅ shipped |
| 04 | `04_barrier.py` | Signals only: `notify(AtomicAdd)`/`wait(Ge)`; single-rendezvous N-rank barrier; reveal `pld.tensor.barrier` | ✅ shipped |
| 05 | `05_remote_load_store.py` | Tile-level RMA: `remote_load`/`remote_store`; one-step ring shift | ✅ shipped |
| 06 | `06_put_get.py` | Tensor-level p2p: `put`/`get`; push vs pull | ✅ shipped |
| 07 | `07_dynamic_rank_count.py` | Dynamic rank count: `pl.dynamic("NR")`; one source, any P | ✅ shipped |
| 08 | `08_allreduce_mesh.py` | All-reduce v1 (mesh): every rank reads every peer, sums locally | ✅ shipped |
| 09 | `09_allreduce_two_phase.py` | All-reduce v2: reduce-scatter + all-gather | ✅ shipped |
| 10 | `10_allreduce_ring.py` | All-reduce v3 (ring): chunked around the ring | ✅ shipped |
| 11 | `11_allreduce_reveal.py` | **The reveal**: `pld.tensor.allreduce` (mesh + ring); diff the IR | ✅ shipped |
| 12 | `12_broadcast.py` | One-to-all; reveal `pld.tensor.broadcast` | planned |
| 13 | `13_allgather.py` | All-to-all slices; reveal `pld.tensor.allgather` | planned |
| 14 | `14_reduce_scatter.py` | All-to-chunks; reveal `pld.tensor.reduce_scatter` | planned |
| 15 | `15_all_to_all.py` | Personalized exchange; reveal `pld.tensor.all_to_all` | planned |
| 16 | `16_putting_it_together.py` | Compose `broadcast` + `allreduce` + `allgather` in one kernel | planned |

Steps 12–16 are **planned** — they arrive in later PRs. The walkthroughs below
(06–16) cover steps 01–11.

## The abstractions map

Every `pld` abstraction: one-line purpose, the chapter section that documents
it, and the tutorial step that teaches it. The **coverage contract** for the
tutorials: nothing exists in code without being teachable from an example.

### System substrate

| Abstraction | Purpose | Chapter section | Runs on | Tutorial step |
| ----------- | ------- | --------------- | ------- | ------------- |
| `pld.world_size()` | Number of ranks in the world | [02-primitives](02-primitives.md) §System Substrate | Host (orchestrator) | 01 |
| `pld.get_comm_ctx(dt)` | Resolve the comm context a `DistributedTensor` belongs to | [02-primitives](02-primitives.md) §System Substrate | Host / InCore | 04 |
| `pld.rank(ctx)` | This rank's index in the context | [02-primitives](02-primitives.md) §System Substrate | InCore | 04 |
| `pld.nranks(ctx)` | Rank count in the context | [02-primitives](02-primitives.md) §System Substrate | InCore | 04 |
| `pl.dynamic("NR")` | Name a runtime-resolved dimension (e.g. the rank count) | [00-getting_started](../00-getting_started.md) | — | 07 |

### Memory

| Abstraction | Purpose | Chapter section | Tutorial step |
| ----------- | ------- | --------------- | ------------- |
| `pld.DistributedTensor` | Window-bound tensor type, visible to peers | [00-model](00-model.md) §Glossary | 03 |
| `pld.alloc_window_buffer(...)` | Allocate a symmetric per-rank window buffer | [02-primitives](02-primitives.md) §Window Buffer Management | 03 |
| `pld.window(...)` | A `DistributedTensor` view of a window buffer | [02-primitives](02-primitives.md) §Window Buffer Management | 03 |

### Signals

| Abstraction | Purpose | Chapter section | Runs on | Tutorial step |
| ----------- | ------- | --------------- | ------- | ------------- |
| `pld.system.notify(...)` | Increment a signal cell on a peer | [02-primitives](02-primitives.md) §Notify & Wait | InCore | 04 |
| `pld.system.wait(...)` | Block until a signal cell reaches a threshold | [02-primitives](02-primitives.md) §Notify & Wait | InCore | 04 |
| `pld.NotifyOp.AtomicAdd` | Notify mode that accumulates contributions (multi-writer safe) | [02-primitives](02-primitives.md) §Choosing NotifyOp and WaitCmp | — | 04 |
| `pld.WaitCmp.Ge` | Wait mode: pass when `>= expected` | [02-primitives](02-primitives.md) §Choosing NotifyOp and WaitCmp | — | 04 |

### Tile-level RMA

| Abstraction | Purpose | Chapter section | Tutorial step |
| ----------- | ------- | --------------- | ------------- |
| `pld.tile.remote_load(...)` | Pull a peer's window slice into a local tile | [02-primitives](02-primitives.md) §Tile-Level RMA | 05 |
| `pld.tile.remote_store(...)` | Push a local tile into a peer's window slice | [02-primitives](02-primitives.md) §Tile-Level RMA | 05 |

### Tensor-level point-to-point

| Abstraction | Purpose | Chapter section | Tutorial step |
| ----------- | ------- | --------------- | ------------- |
| `pld.tensor.put(...)` | Push a local window slice into a peer's window | [02-primitives](02-primitives.md) §Put and Get | 06 |
| `pld.tensor.get(...)` | Pull a peer's window slice into local memory | [02-primitives](02-primitives.md) §Put and Get | 06 |
| `pld.AtomicType` | Put/get atomicity mode | [02-primitives](02-primitives.md) §Put and Get | 06 |

### Collectives

| Abstraction | Purpose | Chapter section | Tutorial step |
| ----------- | ------- | --------------- | ------------- |
| `pld.tensor.barrier(...)` | Synchronize all ranks (revealed builtin) | [01-collectives](01-collectives.md) §Barrier | 04 |
| `pld.tensor.allreduce(...)` | Reduce and broadcast the result (mesh/ring) | [01-collectives](01-collectives.md) §AllReduce | 11 |
| `pld.tensor.broadcast(...)` | One rank's data to all | [01-collectives](01-collectives.md) §Broadcast | 12 |
| `pld.tensor.allgather(...)` | All ranks' slices to all | [01-collectives](01-collectives.md) §AllGather | 13 |
| `pld.tensor.reduce_scatter(...)` | Reduced result, one chunk per rank | [01-collectives](01-collectives.md) §ReduceScatter | 14 |
| `pld.tensor.all_to_all(...)` | Personalized exchange | [01-collectives](01-collectives.md) §AllToAll | 15 |

### Composition

| Abstraction | Purpose | Chapter section | Tutorial step |
| ----------- | ------- | --------------- | ------------- |
| `@pl.jit.host` | Host orchestrator: allocates windows, dispatches ranks | [00-model](00-model.md) §Glossary | 02 |
| `@pl.jit` / `@pl.jit.incore` | Per-device orchestration / device-side kernel | [03-execution](03-execution.md) | 02 |
| `device=r` | Pin one dispatch to one device from the host loop | [00-model](00-model.md) | 01 |
| `DistributedConfig` | Device list + worker count for compilation | [03-execution](03-execution.md) | 01 |

## See also

- [00-model](00-model.md) — Quickstart and model vocabulary
- [01-collectives](01-collectives.md) — The collectives (steps 08–16)
- [02-primitives](02-primitives.md) — The substrate beneath the collectives
- Next step: [06-hello_rank](06-hello_rank.md) — run your first 2-rank program
