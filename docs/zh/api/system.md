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

实际效果取决于后端传输路径。A5 的直接 Acc→UB C2V 管道返回当前
AIV 的本地 UB 槽；显式 pop lane 不会把该槽重定位到另一个物理 AIV。
该路径应使用匹配的物理 lane。当前固定版本的 GM-backed C2V 实现将
lane 用于 GM 数据分带偏移；该路径仍需单独进行真机验证。

PTO-ISA CPU 模拟器尚未实现显式 lane 重载。这些端点在 `__CPU_SIM`
下报告编译期诊断；不会删除 lane 来选择其他重载。NPU 执行与
PyTorch 参考 FIFO 属于独立路径。

`tests/st/runtime/cross_core/test_explicit_aiv_lane.py` 使用精确 golden
检查省略、静态和动态 lane，并通过四槽环形缓冲完成十二次传输。
A5 V2C 用例交换逻辑 lane，区分显式选择与忽略操作数的行为；本地
C2V 用例使用匹配的物理 lane。A3、A5 的运行结果必须分别记录。

## L1 到 UB 的支持边界

显式 lane 不建立 Mat（L1）到 Vec（UB）的传输能力。当前 PTOAS 的
`verifyTMovGenericPairing` 不包含 Mat→Vec，950 的内存图
`src/backend/common/soc.cpp` 也没有该边。在出现合法后端机制的证据前
继续维持限制。

各候选机制需要独立契约：

- 直接 TMOV 需要合法的 Mat→Vec 指令、布局、dtype、有效范围及
  生产者/消费者次序。最小汇编拒绝用例位于
  `tests/ut/codegen/test_pto_codegen_aiv_lane.py`。
- Tile TPipe 需要证明 C2V 生产端接受 Mat、AIV 消费端得到 Vec；
  已有 Acc→Vec 管道不能证明这一点。当前 Mat 生产端的最小用例在
  A3、A5 均报错 `tile type must map to a supported producer pipe`。
- 经 GM 路径需要合法的 L1→GM store、AIV GM→UB load、生命周期、cache
  可见性和同步。A3 store verifier 允许 Mat，A5 仅允许 Vec 或 Acc。
  当原始 GM 输入仍可用时，可以单独编写在 AIV 重新读取的算法，
  但这不等于复制已有的任意 L1 tile。

本次不添加内存图边或隐式 staging。这些编译器限制不能证明硬件没有相关
能力。扩展该路径仍需后端指令和协议证据，并明确外部 L1→UB 需求所指机制。

::: pypto.language.system
    options:
      show_root_heading: false
      filters: ["!^_"]
