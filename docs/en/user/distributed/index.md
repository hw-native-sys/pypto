# Distributed Programming

PyPTO's distributed model is built on **symmetric memory and signals**:
every rank sees the same window-buffer address across peers, reaches other
ranks through one-sided `put`/`get`/`remote_load`, and coordinates through
**signal synchronisation** (`notify`/`wait`). A **comm domain** is a subset
of ranks sharing a symmetric window pool; the full world is the default
domain.

Every allreduce, broadcast, and barrier the compiler lowers is a composition
of these same primitives — the `pld.tensor.*` collectives (`allreduce`,
`barrier`, etc.) are syntactic sugar over them, not a separate library.

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
[Language Guide](../01-language_guide.md).

## Glossary

| Term | Definition |
| ---- | ---------- |
| **Rank** | A single process or chip participating in a distributed program. Each rank has a unique rank index assigned at launch time. |
| **Device** | One Ascend NPU chip (or die), identified by a `device_id`. One rank maps to one device. |
| **Node** | A physical machine hosting one or more devices. |
| **Window buffer** | A symmetric per-rank HCCL buffer. Ranks see peers through `CommContext.windowsIn[peer]`/`windowsOut[peer]`. |
| **Comm domain** | A subset of ranks sharing a symmetric window pool. Default: the full world. |
| **Signal** | A cross-rank synchronisation primitive. Notify/wait counters coordinate access to window buffers. |
| **Orchestrator** | The HOST function that allocates window buffers and dispatches kernels to devices. |
| **InCore kernel** | The device-side function that executes on the NPU. |

## Reading Path

1. **[00-model](00-model.md)** — Quickstart-first: run a 2-rank program, then the model vocabulary
2. **[01-collectives](01-collectives.md)** — AllReduce, barrier, broadcast, allgather, reduce_scatter, all-to-all
3. **[02-primitives](02-primitives.md)** — notify/wait, remote_load/remote_store, put/get, CommCtx
4. **[03-execution](03-execution.md)** — DistributedWorker lifecycle, DeviceTensor, multi-program, env vars
5. **[04-debugging](04-debugging.md)** — Common failure patterns and diagnostic flags

## See Also

- [Getting Started](../00-getting_started.md) — `ir.compile()`, `CompiledProgram`, `DeviceTensor`, `RunConfig`
- [Simpler Runtime](https://hw-native-sys.github.io/simpler/) — Runtime internals (scheduler, graph building, tensormap)
