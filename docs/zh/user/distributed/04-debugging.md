# 调试与陷阱

分布式 bug 很少留下本地堆栈——症状出现在某个 rank 上，而原因却在另一个
rank 上。

本页是本章**所有分布式陷阱的权威索引**。下面的常见故障模式和致命陷阱是
章节级的（模型、原语、集合通信）；教程阶梯的每个步骤还各自有一节更窄的
"边界情况"，覆盖该步骤算法特有的 bug——完整交叉引用见
[分步陷阱](#分步陷阱)。

## 常见故障模式

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **所有 rank 挂起** | notify/wait 顺序错误 | 确保 notify 循环在 wait 循环之前。若某 rank 的 `wait` 与它自己的 send 分属不同派发，也可用 `pl.submit(..., deps=[send_task])` 显式钉住顺序——见[声明一条边](../tasks/02-submit.md)。 |
| **静默数据损坏** | `remote_load` offsets 或 shape 不匹配 | 验证 offsets 与对端的 store offsets 对齐。 |
| **Signal cell 永不达到期望值** | 错误 `NotifyOp` | 多参与者屏障用 `AtomicAdd`；1:1 交换用 `Set`。 |
| **编译时形状不匹配** | `NR` 未使用 `pl.dynamic` | 将运行时维度包裹在 `pl.dynamic("NR")` 中。 |
| **派发时抛出 `TypeError`** | IO buffer 在 `prepare()` 前未调用 `.share_memory_()`——fork 出的子进程看不到 fork 之后分配的 buffer | 在 `prepare()` 之前对每个传给 worker 的 host tensor 调用 `.share_memory_()`。 |

## 致命陷阱

> **缺少 `.share_memory_()`：** 传入 `DistributedWorker` 的 IO buffer 在
> `prepare()` 前必须调用 `.share_memory_()`。若忘记调用，运行时会在派发时
> 抛出 `TypeError`——fork 出的子进程无法访问父进程私有的内存。
>
> **`alloc_window_buffer` 传入 rank 数量而非字节数：** `size` 参数以**字节**
> 为单位。使用 shape+dtype 重载。
>
> **分发循环的执行次数与 `device_ids` 不匹配：** `device_ids` 是物理卡 ID
>（例如来自 `--device 4,5`），不要求从 0 开始或连续。`device=r` 则是一个
> *逻辑* rank 索引，始终按 `[0, world)` 校验，其中
> `world = len(device_ids)`；运行时按 `rank r -> device_ids[r]` 映射。真正
> 重要的不变量是分发循环的执行次数等于 `len(device_ids)`——写成
> `for r in pl.range(pld.world_size())` 时会自动满足。不匹配的例子：
> `device_ids=[0, 1, 2, 3]`（4 张卡）但分发循环只覆盖 `range(2)`，会让
> 2 张卡未被派发，导致未定义行为（`MaterializeCommDomainScopes` 要求
> `device=r` 循环的范围必须是 `[0, N)`）。

## 分步陷阱

教程阶梯的每个步骤都有自己的"边界情况"一节，覆盖该步骤算法特有的 bug。
下表是全部十六个步骤的统一索引——上面的章节级模式对每个步骤都通用：

| 步骤 | 页面 | 致命陷阱 |
| ---- | ---- | -------- |
| 01 | [06-hello_rank](06-hello_rank.md) | 标量参数排在张量参数之前——`TaskArgs: cannot add tensor after scalar` |
| 02 | [07-programming_model](07-programming_model.md) | 在主机分发循环之外读取 rank 身份，会让所有 rank 得到相同的值 |
| 03 | [08-window_buffer](08-window_buffer.md) | 把 window 绑定的 `DistributedTensor` 当作普通 `Tensor`（或反之）——编译期类型错误 |
| 04 | [09-barrier](09-barrier.md) | 在共享单元 barrier 上使用 `Set`/`Eq`，会静默覆盖更早到达的对端 |
| 05 | [10-remote_load_store](10-remote_load_store.md) | 在排序 barrier 之前做 RMA——读取对端尚未 staging 好的 window 内存 |
| 06 | [11-put_get](11-put_get.md) | `put` 不配对 notify/wait——读取会与传输竞争 |
| 07 | [12-dynamic_rank_count](12-dynamic_rank_count.md) | 主机形状里留下写死的 rank 数量，使 `pl.dynamic("NR")` 失效 |
| 08 | [13-allreduce_mesh](13-allreduce_mesh.md) | 缺少 barrier 让读取与 store 竞争——与时序相关，可能 P=2 通过而 P=4 失败 |
| 09 | [14-allreduce_two_phase](14-allreduce_two_phase.md) | 两个 barrier 复用同一行信号——单调计数器让第二个 barrier 提前返回 |
| 10 | [15-allreduce_ring](15-allreduce_ring.md) | rank 0 处左邻居索引在截断取模下取负 |
| 11 | [16-allreduce_reveal](16-allreduce_reveal.md) | 内置原语的信号形状与模式（`ring` 还是 `mesh`）不匹配 |
| 12 | [17-broadcast](17-broadcast.md) | 从 `my_rank` 自己的 slice 而非根的 slice 广播 |
| 13 | [18-allgather](18-allgather.md) | gather 到错误的输出槽位——偏移与对端 rank 不匹配 |
| 14 | [19-reduce_scatter](19-reduce_scatter.md) | 归约时遗漏自己的 window 行 |
| 15 | [20-all_to_all](20-all_to_all.md) | 源和结果复用同一个 window 缓冲 |
| 16 | [21-putting_it_together](21-putting_it_together.md) | 让同一个共享 window 对应两种不同的信号布局（mesh 与 ring） |

## 诊断标志

`SIMPLER_HOST_STRACE` 和 `SIMPLER_DFX` 是**编译时 C 预处理器宏**，设置为 shell
环境变量**无效**——它们在编译期就已固定。默认开启。切换它们属于 `simpler`
运行时的构建配置变更，而非一个简单的 `cmake -D...` 缓存变量——具体机制见
`simpler` 运行时自己的构建文档。

运行时环境变量：

```bash
# 切换设备域 [STRACE] 标记：
SIMPLER_DEVICE_STRACE_ENABLE=0 python script.py
```

### 分布式 DFX 入口点

- **L2 swimlane：** `RunConfig(enable_chip_swimlane=True)`——在 worker 内部启用
  逐任务计时，并透传到 L3 编排。写入 `dfx_outputs/chip_swimlane_records.json`
  （onboard 场景会与下面的依赖图合并为 `merged_swimlane_*.json`）。
- **Scope 统计：** `RunConfig(enable_scope_stats=True)`——写入
  `dfx_outputs/scope_stats/scope_stats.jsonl`，包含 task_window、heap 和
  tensormap 水位。
- **依赖图：** `RunConfig(enable_dep_gen=True)`——写入 `dfx_outputs/deps.json`，
  供调度器分析的任务依赖图。

## 相关链接

- [00-model](00-model.md) — 快速开始和模型词汇
- [02-primitives](02-primitives.md) — 集合通信的底层基础
