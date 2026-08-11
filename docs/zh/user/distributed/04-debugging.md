# 调试与陷阱

分布式 bug 很少留下本地堆栈——症状出现在某个 rank 上，而原因却在另一个
rank 上。

## 常见故障模式

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **所有 rank 挂起** | notify/wait 顺序错误 | 确保 notify 循环在 wait 循环之前。 |
| **静默数据损坏** | `remote_load` offsets 或 shape 不匹配 | 验证 offsets 与对端的 store offsets 对齐。 |
| **Signal cell 永不达到期望值** | 错误 `NotifyOp` | 多参与者屏障用 `AtomicAdd`；1:1 交换用 `Set`。 |
| **编译时形状不匹配** | `NR` 未使用 `pl.dynamic` | 将运行时维度包裹在 `pl.dynamic("NR")` 中。 |
| **派发时抛出 `TypeError`** | IO buffer 在 `prepare()` 前未调用 `.share_memory_()`——fork 出的子进程看不到 fork 之后分配的 buffer | 在 `prepare()` 之前对每个传给 worker 的 host tensor 调用 `.share_memory_()`。 |
| **循环内 allreduce 被拒绝** | Signal 协议无法每轮注入新 buffer | 在循环外每次调用分配新 signal buffer。 |

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

- **L2 swimlane：** `RunConfig(enable_l2_swimlane=True)`——在 worker 内部启用
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
