# 块内跨核同步（intra-block synchronization）

## 事件形式与 pipe

这组操作用于同一执行块（block）内 AIC（Cube）与 AIV（Vector）之间的同步。
`pl.system.set_intra_block` 发出同步事件，`pl.system.wait_intra_block` 等待对应
`event_id` 的事件。二者分别发射 PTOAS 的 `pto.set_intra_block` 和
`pto.wait_intra_block`，均无返回值；现有 `sync_set` 的 FFTS mode 0/1/2 契约保持不变。
A5 静态 ID 范围为 **0–31**，动态 ID 必须是
`pl.Scalar[pl.INT32]` 或 `pl.Scalar[pl.INT64]`。DSL 整数字面量仍保存为静态属性；
动态 INDEX 标量需要显式转换；
编译器不插入隐式转换或动态范围检查。A5 合法 pipe 为 `FIX`、`MTE1`、
`MTE2`、`MTE3`、`V` 和 `S`，拒绝 `M` 和 `ALL`。`S` 还要求 PTOAS 的 A5 块内同步事件校验支持该 pipe；
旧编译器可能拒绝已发射的 PTO。set/wait 可使用不同的
合法 pipe：各自约束本核的生产/消费流水线，事件标识会合关系。
A2/A3 的 PTOAS 路径使用 FFTS mode 2，ID 限于 0–15，且不支持 `S`；
不能将 A5 的范围或单 AIV 语义推广到这些目标。

```python
# Placement example only, not a complete handshake.
# The wait runs on AIV1; select that lane in the surrounding kernel.
pl.system.set_intra_block(17, pipe=pl.PipeType.MTE1, core_type=pl.KernelType.AIC)
pl.system.wait_intra_block(1, pipe=pl.PipeType.V, core_type=pl.KernelType.AIV)
```

## 流水线顺序

当 `pl.pipeline` 循环体包含块内同步事件时（包括嵌套控制流中的事件），
无论 `stage` 取值多少，编译器均保守地将其降为顺序循环。
这保留了手写的 wait → 数据访问 → set 顺序，以及跨迭代的事件复用约束。
块内同步事件与数据访问之间没有 SSA 依赖，IO 重排和软件流水不能越过该握手；
编译器不会额外插入同步指令。

## 核归属与 AIV 事件编号

`core_type` 选择 AIC 或 AIV，**不选择 AIV0 或 AIV1**。混合核中应显式指定；
展开时拒绝未指定归属的块内同步事件，避免复制其副作用。在明确标为 AIC/AIV 的
函数中可以省略。Python API 不提供 lane 参数。本机 A5 PTO-ISA 的
`TPush.hpp` 使用 16 个本地事件和 `VEC_CORE_ID_OFFSET = 16`：对于
0–15 的本地事件 `e`，Cube 用 `e` 寻址 AIV0，用 `e + 16` 寻址 AIV1；
各 AIV 使用本地 `e`。Cube 必须分别向参与的 AIV 发信号/等待，因此 Cube
上的原始事件 17 对应 AIV1 的本地事件 1，而不是两个 AIV 的本地事件 17。
仅在一个 AIV 执行还需要核函数中的 lane 控制。配套单 AIV 真机协议在
Cube 端通过 MTE1 等待和 `PIPE_ALL` 栅栏约束事件复用，防止标量发射越过
回执。块内同步 `PIPE_S` 的编译器支持仍单独检查。

## 资源归属

这些调用不预留事件。TPipe 使用同一个 intra-block 命名空间：
单向 pipe 消耗 `flag_base` 和 `flag_base + 1`，双向 pipe 还消耗 `+2` 和
`+3`，Cube 端还有 AIV1 的 `+16` 别名。PTOAS 分配器仅统计 pipe 组件，
不统计手写块内同步事件。必须避开所有存活的 pipe 区间及其别名，以及混合
`SYNCALL` 使用的事件（以目标 PTO-ISA 常量为准；本机 A5 头文件使用本地 12/13 和 Cube
端别名 28/29）。FFTS API 的 14/15 预留
并不意味着块内同步事件可以自由使用这两个编号。自动核内 `set_flag`/`wait_flag`
使用独立命名空间。协议作者负责分配不冲突的 ID、平衡各轮 set/wait，
并保证负载可见性和缓冲区复用顺序。

## 原生负载验证

复用缓冲区验收用例通过 FIX 将 Cube Acc tile 写入 GM，AIV 通过 MTE2
等待并读取负载，再通过 MTE3 回执保护下一次 Cube 覆盖。
两条 AIV lane 均使用精确的迭代标签验证 ready/ack 顺序。
显式 `CORE_GROUP` 共同调度 Cube producer 和按 lane 选择的 AIV consumer，
使手写 GM 协议与混合核中的自动 direct C2V promotion 分离。
该 promotion 与手写 FFTS 事件一样，也拒绝与手写块内同步事件混用的 GM 交换；
手写协议应使用显式 peer kernel。
这些 DMA 路径的证据不能推广为标量 D-cache 到 DMA 的发布语义保证。

## Debug 后端支持范围

Torch debug 代码生成会拒绝包含这些操作的程序：其混合核多线程运行时
尚未对块内同步事件的顺序和 AIV 事件编号建模。此类程序应使用原生 PTO 后端。
静默省略会合操作可能导致消费者在生产者写入共享负载之前就进行读取。
