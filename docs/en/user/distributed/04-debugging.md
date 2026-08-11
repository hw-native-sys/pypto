# Debugging and Pitfalls

Distributed bugs rarely leave a local stack trace — the symptom shows up on
one rank while the cause is on another.

## Common Failure Patterns

| Symptom | Likely Cause | Fix |
| ------- | ------------ | --- |
| **All ranks hang** | Notify/wait ordering — a rank is waiting on a peer that hasn't notified yet | Ensure every rank calls `notify` before any rank calls `wait`. The notify loop should precede the wait loop. |
| **Silent data corruption** | `remote_load` offsets or shape don't match what the peer stored | Verify offsets align with the peer's store offsets. A 1-element shift introduces a full row of garbage. |
| **Signal cell never reaches expected value** | Wrong `NotifyOp`: used `Set` instead of `AtomicAdd` for a multi-participant barrier | Use `AtomicAdd` when N ranks contribute to the same slot; use `Set` for 1:1 exchanges. |
| **Shape mismatch at compile time** | `NR` (world size) used in type annotations without `pl.dynamic` | Wrap runtime-resolved dims in `pl.dynamic("NR")`. The compiler needs the name to bind the runtime value. |
| **`TypeError` raised at dispatch** | IO buffer not `.share_memory_()` before `prepare()` — the child processes cannot see a buffer allocated after the fork | Call `.share_memory_()` on every host tensor passed to the worker, before `prepare()`. |
| **Allreduce rejected inside loop** | Signal protocol can't inject a fresh buffer per iteration | Allocate a fresh signal buffer for each allreduce call outside loops; allreduce inside `for`/`while` is currently rejected. |

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

- **L2 swimlane:** `RunConfig(enable_l2_swimlane=True)` — enables per-task timing
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
