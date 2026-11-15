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

## 显式 AIV lane

`tpush_to_aic` 和 `tpop_from_aic` 接受可选关键字 `aiv_subblockid`。
Python 整数 0 或 1 转为 INT64 常量；动态操作数必须为
`pl.Scalar[pl.INT64]`。INDEX 或其他标量类型需要通过
`pl.cast(value, pl.INT64)` 显式转换。lane 保存在 SSA 操作数中，经过
outlining（函数提取）和拆核后仍保留。省略参数时维持后端物理 lane
选择和原有管道发射行为。

```python
lane = pl.cast(pl.tile.get_subblock_idx(), pl.INT64)
received = pl.system.tpop_from_aic(
    shape=[16, 16], dtype=pl.FP32, split=1, aiv_subblockid=lane
)
pl.system.tpush_to_aic(received, split=1, aiv_subblockid=lane)
pl.system.tfree_to_aic(received)
```

该操作数仅适用于 AIV 侧 Vec Tile entry 和 split=1/2/3/4。
no-split（`split=0`）、GlobalTensor entry 以及 AIC 侧的
`tpush_to_aiv` / `tpop_from_aiv` 不接受该参数。已有 split、形状、布局和
初始化契约仍然有效；该参数不创建管道，也不修改 init schema。
特别是 PyPTO 继续拒绝 V2C push 的 odd split 3/4。
该矩阵描述编译器契约；设备执行需要分别验证每个架构。

| 路径 | A3 | A5 |
| ---- | -- | -- |
| 显式 lane，Vec push/pop，split 1/2 | PTOAS 支持该操作数 | PTOAS 支持该操作数；保留 V2C 布局适配 |
| 显式 lane，C2V pop，split 3/4 | 遵守已有 GM-backed odd split 规则 | 遵守已有 GM-backed odd split 规则 |
| 显式 lane，V2C push，split 3/4 | 保留 PyPTO 限制 | 后端拒绝 odd V2C |
| no-split / GlobalTensor / AIC 侧 lane | 拒绝 | 拒绝 |

运行时由作者保证动态 lane 属于 `{0, 1}`，按匹配的 split 协议提供各个
逻辑 lane，并配对 push/pop/free 次数与 pipe ID。不插入运行时范围检查；
越界常量会被拒绝。逻辑 lane 的选择不会把操作迁移到其他物理核，
也不会改变执行它的 kernel。

::: pypto.language.system
    options:
      show_root_heading: false
      filters: ["!^_"]
