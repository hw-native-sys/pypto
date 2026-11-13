# `pl.system`

同步、cache 与跨核原语。它们是产生副作用的语句而非产值算子 —— 多数没有返回值。

流水线屏障统一使用 `pl.system.barrier(pipe=...)`，接受 `pl.PipeType` 的任意成员。
屏障按流水生效而非按算子生效：它排空该流水上已发出的一切，因此一次搬运由哪条
流水承载，就由哪条屏障为它定序 —— `MTE2` 对应 GM → UB/L1 搬入，`MTE3` 对应
UB → GM 搬出以及 A5 的 UB → L1 路由，`MTE1` 对应 L1 → L0A/L0B 的操作数喂入，
`FIX` 对应 L0C 写回。`ALL` 覆盖当前核的所有流水线；跨核到达同步使用
`pl.system.syncall(...)`。
原有 `bar_v()`、`bar_m()`、`bar_all()` 接口已移除。具有源、目标方向的事件
交接仍使用独立的 `set_flag` / `wait_flag` 配对，参见
[流水线定序](../user/language/04-scopes.md#跨-lane-的定序同样由你负责)。

::: pypto.language.system
    options:
      show_root_heading: false
      filters: ["!^_"]
