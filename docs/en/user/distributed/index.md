# Distributed Programming

PyPTO's distributed model is built on **symmetric memory and signals** — see
[00-model](00-model.md) for the full explanation. In short: every rank's
window buffer has the same layout as every peer's, reaches other ranks
through one-sided `put`/`get`/`remote_load`, and coordinates through
**signal synchronisation** (`notify`/`wait`).

Every allreduce, broadcast, and barrier the compiler lowers is a composition
of these same primitives — the `pld.tensor.*` collectives (`allreduce`,
`barrier`, etc.) are syntactic sugar over them, not a separate library.

## Symmetric Memory at a Glance

```text
                    comm domain (default: full world)
   ┌───────────────────────┬───────────────────────┬───────────────────────┐
   │        rank 0         │        rank 1         │        rank 2         │
   │  window base: 0x1000  │  window base: 0x5000  │  window base: 0x9000  │
   └───────────────────────┴───────────────────────┴───────────────────────┘
       ▲ every rank's own base can differ. Each rank also keeps a lookup
         table, windowsIn[peer], mapping every peer's rank index to that
         peer's base — so rank 1 reaches rank 0's data as
         windowsIn[0] + offset, not a shared absolute pointer. "Symmetric"
         means every window has the same size and layout (offset X is the
         same slice on every rank), never that ranks share one address.
```

Every rank's window has the same layout — that is the substance of
"symmetric" — but each rank's own base address can differ. Every rank keeps
a lookup table (`CommContext.windowsIn[peer]`) of every peer's base, so
`remote_load`/`remote_store`/`put`/`get` compute `windowsIn[peer] + offset`
locally rather than asking the peer for its address. Signals (`notify`/
`wait`) are the separate mechanism that tells a rank *when* that data is
ready to read — the layout alone doesn't guarantee that.

## L2 vs L3

| Layer | Scope | API namespace |
| ----- | ----- | ------------- |
| L2 | Single-device (one NPU chip) | `pl.*` |
| L3 | Cross-rank (multiple NPUs or processes) | `pld.*` |

> **PyPTO's L2/L3 vs simpler's L0–L6:** these two tiers are PyPTO's own
> user-facing vocabulary, not simpler's numbering. Simpler uses a finer
> seven-level hierarchy (L0 core → L1 die → L2 chip → L3 host → L4 pod →
> L5 super-node → L6 cluster); PyPTO's "L2" spans simpler's L0–L2 (everything
> on one chip), and PyPTO's "L3" spans simpler's L3 and up (everything across
> chips). See simpler's
> [Hierarchical Level Runtime](https://hw-native-sys.github.io/simpler/hierarchical-level-runtime/)
> for the full model.

The distributed chapter covers L3. L2 is covered in the
[Compiling](../execution/00-compile.md).

## Glossary

| Term | Definition |
| ---- | ---------- |
| **Rank** | A single process or chip participating in a distributed program. Each rank has a unique rank index assigned at launch time. |
| **Device** | One Ascend NPU chip (or die), identified by a `device_id`. One rank maps to one device. |
| **Node** | A physical machine hosting one or more devices. |
| **Symmetric memory** | The property that, within a communication domain, every rank's window buffer has the same size and layout — each rank reaches a peer's data via its own `windowsIn[peer]` lookup plus a local offset, not a shared absolute address. See [Symmetric Memory at a Glance](#symmetric-memory-at-a-glance) above. |
| **Window buffer** | A symmetric per-rank HCCL buffer. Ranks see peers through `CommContext.windowsIn[peer]`/`windowsOut[peer]`. |
| **Window buffer address space** | The address range a window buffer occupies within one rank. Every rank's window has the same size and layout within a comm domain — not the same absolute address — which is what makes the buffer "symmetric." |
| **Comm domain** | A subset of ranks sharing a symmetric window pool. Default: the full world. |
| **Signal** | A cross-rank synchronisation primitive. Notify/wait counters coordinate access to window buffers. |
| **Orchestrator** | The HOST function that allocates window buffers and dispatches kernels to devices. |
| **InCore kernel** | The device-side function that executes on the NPU. |

## Reading Path

1. **[00-model](00-model.md)** — Quickstart-first: run a 2-rank program, then the model vocabulary
2. **[01-collectives](01-collectives.md)** — AllReduce, barrier, broadcast, allgather, reduce_scatter, all-to-all
3. **[02-primitives](02-primitives.md)** — notify/wait, remote_load/remote_store, put/get, CommCtx
4. **[03-execution](03-execution.md)** — DistributedWorker lifecycle, DeviceTensor, multi-program, env vars
5. **[04-debugging](04-debugging.md)** — Common failure patterns, diagnostic flags, and the per-step pitfall index
6. **[05-tutorials](05-tutorials.md)** — The 16-step runnable tutorial ladder (`examples/distributed/`); walkthrough page `NN` teaches example `(NN-5)_*.py` (e.g. `06-hello_rank.md` walks through `01_hello_rank.py`)

## See Also

- [Getting Started](../00-getting_started.md) — `ir.compile()`, `CompiledProgram`, `DeviceTensor`, `RunConfig`
- [Simpler Runtime](https://hw-native-sys.github.io/simpler/) — Runtime internals (scheduler, graph building, tensormap)
