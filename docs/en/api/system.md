# `pl.system`

Synchronization, cache and cross-core primitives. These are side-effecting statements
rather than value-producing operators — most return nothing.

Use `pl.system.barrier(pipe=...)` with `pl.PipeType.V`, `pl.PipeType.M`, or
`pl.PipeType.ALL` for a pipeline barrier. `ALL` covers all pipelines on the
current core; cross-core arrival uses `pl.system.syncall(...)`.
The old `bar_v()`, `bar_m()`, and `bar_all()` APIs have been removed.
Directional event handoffs still use a separate
`set_flag` / `wait_flag` pair; see [pipeline ordering](../user/language/04-scopes.md#cross-lane-ordering-is-yours-too).

::: pypto.language.system
    options:
      show_root_heading: false
      filters: ["!^_"]
