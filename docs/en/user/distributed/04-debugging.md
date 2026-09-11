# Debugging and Pitfalls

Distributed bugs rarely leave a local stack trace — the symptom shows up on
one rank while the cause is on another.

This page is the **canonical index of every distributed pitfall** in this
chapter. The failure patterns and fatal pitfalls below are chapter-wide
(model, primitives, collectives); each tutorial-ladder step additionally has
its own narrower "Edge cases" section for bugs specific to that step's
algorithm — see [Per-Step Pitfalls](#per-step-pitfalls) for the full
cross-reference.

## Common Failure Patterns

| Symptom | Likely Cause | Fix |
| ------- | ------------ | --- |
| **All ranks hang** | Notify/wait ordering — a rank is waiting on a peer that hasn't notified yet | Ensure every rank calls `notify` before any rank calls `wait`. The notify loop should precede the wait loop. When a rank's `wait` lives in a different dispatch from its own send, you can also pin the order explicitly with `pl.submit(..., deps=[send_task])` — see [Declaring an Edge](../tasks/02-submit.md). |
| **Silent data corruption** | `remote_load` offsets or shape don't match what the peer stored | Verify offsets align with the peer's store offsets. A 1-element shift introduces a full row of garbage. |
| **Signal cell never reaches expected value** | Wrong `NotifyOp`: used `Set` instead of `AtomicAdd` for a multi-participant barrier | Use `AtomicAdd` when N ranks contribute to the same slot; use `Set` for 1:1 exchanges. |
| **Shape mismatch at compile time** | `NR` (world size) used in type annotations without `pl.dynamic` | Wrap runtime-resolved dims in `pl.dynamic("NR")`. The compiler needs the name to bind the runtime value. |
| **`TypeError` raised at dispatch** | IO buffer not `.share_memory_()` before `prepare()` — the child processes cannot see a buffer allocated after the fork | Call `.share_memory_()` on every host tensor passed to the worker, before `prepare()`. |

## Fatal Pitfalls

> **Missing `.share_memory_()`:** IO buffers passed to `DistributedWorker` must
> call `.share_memory_()` before `prepare()`. If you forget, the runtime raises
> a `TypeError` at dispatch time — the child processes cannot access the parent's
> private memory.
>
> **`alloc_window_buffer` given a rank count instead of bytes:** The `size`
> argument to `alloc_window_buffer` is **in bytes**, not elements. Calling
> `alloc_window_buffer(NR)` allocates `NR` bytes, not `NR * sizeof(element)`.
> Use the shape+dtype overload: `alloc_window_buffer([NR, SIZE], dtype=pl.FP32)`.
>
> **Dispatch loop trip count disagreeing with `device_ids`:** `device_ids`
> are physical card IDs (e.g. from `--device 4,5`) — they need not start at
> 0 or be contiguous. `device=r` is a *logical* rank index, always
> validated against `[0, world)` where `world = len(device_ids)`; the
> runtime maps `rank r -> device_ids[r]`. What must hold is that the
> dispatch loop's trip count equals `len(device_ids)` — automatic when you
> write `for r in pl.range(pld.world_size())`. A mismatch — e.g.
> `device_ids=[0, 1, 2, 3]` (4 cards) but a dispatch loop that only covers
> `range(2)` — leaves 2 cards un-dispatched and causes undefined behaviour
> (`MaterializeCommDomainScopes` requires the `device=r` loop range to be
> `[0, N)`).

## Per-Step Pitfalls

Each tutorial-ladder step has its own "Edge cases" section covering the bug
specific to that step's algorithm. This table is the single index into all
sixteen — the chapter-wide patterns above are common to every step:

| Step | Page | Fatal pitfall |
| ---- | ---- | ------------- |
| 01 | [06-hello_rank](06-hello_rank.md) | Scalar argument placed before a tensor argument — `TaskArgs: cannot add tensor after scalar` |
| 02 | [07-programming_model](07-programming_model.md) | Reading rank identity outside the host dispatch loop gives every rank the same value |
| 03 | [08-window_buffer](08-window_buffer.md) | Treating a window-bound `DistributedTensor` as a plain `Tensor` (or vice versa) — compile-time type error |
| 04 | [09-barrier](09-barrier.md) | `Set`/`Eq` on a shared-cell barrier silently clobbers earlier arrivals |
| 05 | [10-remote_load_store](10-remote_load_store.md) | RMA before the ordering barrier — reading window memory the peer hasn't staged yet |
| 06 | [11-put_get](11-put_get.md) | `put` without a paired notify/wait — the read races the transfer |
| 07 | [12-dynamic_rank_count](12-dynamic_rank_count.md) | Leaving a hardcoded rank count in the host shape defeats `pl.dynamic("NR")` |
| 08 | [13-allreduce_mesh](13-allreduce_mesh.md) | Missing barrier lets a load race the store — timing-dependent, may pass at P=2 and fail at P=4 |
| 09 | [14-allreduce_two_phase](14-allreduce_two_phase.md) | Reusing one signal row for both barriers — the monotonic counter lets the second barrier return early |
| 10 | [15-allreduce_ring](15-allreduce_ring.md) | Negative left-neighbour index at rank 0 under truncating modulo |
| 11 | [16-allreduce_reveal](16-allreduce_reveal.md) | Wrong signal shape for the builtin's mode (`ring` vs `mesh`) |
| 12 | [17-broadcast](17-broadcast.md) | Broadcasting from `my_rank`'s own slice instead of the root's |
| 13 | [18-allgather](18-allgather.md) | Gathering into the wrong output slot — offset doesn't match the peer's rank |
| 14 | [19-reduce_scatter](19-reduce_scatter.md) | Omitting your own window row from the reduction |
| 15 | [20-all_to_all](20-all_to_all.md) | Reusing one window buffer for both source and result |
| 16 | [21-putting_it_together](21-putting_it_together.md) | Pointing one shared window at two different signal layouts (mesh vs ring) |

## Diagnostic Flags

`SIMPLER_HOST_STRACE` and `SIMPLER_DFX` are **compile-time C preprocessor macros**
(`#define` in `profiling_config.h`), not environment variables. Setting them as
shell env vars (e.g. `SIMPLER_DFX=1 python script.py`) has **no effect** — they
are baked in at build time. They default to `1` (enabled). Flipping them is a
`simpler` runtime build-configuration change, not something set via a bare
`cmake -D...` cache variable — see the `simpler` runtime's own build
documentation for the current mechanism.

Runtime environment variables:

```bash
# Toggle device-domain [STRACE] markers at runtime:
SIMPLER_DEVICE_STRACE_ENABLE=0 python script.py
```

### Distributed DFX Entry Points

- **L2 swimlane:** `RunConfig(enable_chip_swimlane=True)` — enables per-task timing
  inside the worker, propagates through L3 orchestration. Writes
  `dfx_outputs/chip_swimlane_records.json` (onboard: merged into
  `merged_swimlane_*.json` alongside the dependency graph below).
- **Scope stats:** `RunConfig(enable_scope_stats=True)` — writes
  `dfx_outputs/scope_stats/scope_stats.jsonl` with task_window, heap, and tensormap watermarks.
- **Dependency graph:** `RunConfig(enable_dep_gen=True)` — writes
  `dfx_outputs/deps.json`, the task dependency graph for scheduler analysis.

## See Also

- [00-model](00-model.md) — Quickstart and model vocabulary
- [02-primitives](02-primitives.md) — The substrate beneath the collectives
