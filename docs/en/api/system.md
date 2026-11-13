# `pl.system`

Synchronization, cache and cross-core primitives. These are side-effecting statements
rather than value-producing operators — most return nothing.

Use `pl.system.barrier(pipe=...)` with any `pl.PipeType` member for a pipeline
barrier. A barrier is pipe-scoped: it drains whatever was issued on that pipe,
so the pipe a transfer rides decides which barrier orders it — `MTE2` the
GM → UB/L1 loads, `MTE3` the UB → GM stores and the A5 UB → L1 route, `MTE1` the
L1 → L0A/L0B operand feed, and `FIX` the L0C writeback. `ALL` covers all
pipelines on the current core; cross-core arrival uses `pl.system.syncall(...)`.
The old `bar_v()`, `bar_m()`, and `bar_all()` APIs have been removed.
Directional event handoffs still use a separate
`set_flag` / `wait_flag` pair; see [pipeline ordering](../user/language/04-scopes.md#cross-lane-ordering-is-yours-too).

::: pypto.language.system
    options:
      show_root_heading: false
      filters: ["!^_"]
