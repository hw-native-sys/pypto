# `pl.system`

同步、cache 与跨核原语。它们是产生副作用的语句而非产值算子 —— 多数没有返回值。

流水线屏障统一使用 `pl.system.barrier(pipe=...)`，参数可选 `pl.PipeType.V`、
`pl.PipeType.M` 或 `pl.PipeType.ALL`。`ALL` 覆盖当前核的所有流水线；跨核到达
同步使用 `pl.system.syncall(...)`。
原有 `bar_v()`、`bar_m()`、`bar_all()` 接口已移除。具有源、目标方向的事件
交接仍使用独立的 `set_flag` / `wait_flag` 配对，参见
[流水线定序](../user/language/04-scopes.md#跨-lane-的定序同样由你负责)。

::: pypto.language.system
    options:
      show_root_heading: false
      filters: ["!^_"]
