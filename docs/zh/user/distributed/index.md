# 分布式编程

PyPTO 的分布式模型建立在**对称内存与信号**之上：每个 rank 在所有对端看到
相同的 window buffer 地址，通过单边 `put`/`get`/`remote_load` 访问其他
rank，并通过**信号同步**（`notify`/`wait`）进行协调。**通信域**是共享对称
window pool 的 rank 子集；整个 world 为默认通信域。

编译器 lowering 出的每个 allreduce、broadcast、barrier 都是这些相同原语的
组合——`pld.tensor.*` 集合通信（`allreduce`、`barrier` 等）是它们的语法糖，
而非另一套独立的库。

## L2 vs L3

| 层级 | 范围 | API 命名空间 |
| ---- | ---- | ------------ |
| L2 | 单设备（一个 NPU 芯片） | `pl.*` |
| L3 | 跨 rank（多个 NPU 或进程） | `pld.*` |

> **PyPTO 的 L2/L3 与 simpler 的 L0–L6：** 这两层是 PyPTO 自己的用户侧词汇，
> 并非 simpler 的编号体系。simpler 使用更细的七层体系（L0 核心 → L1 die →
> L2 芯片 → L3 主机 → L4 pod → L5 超节点 → L6 集群）；PyPTO 的 "L2" 对应
> simpler 的 L0–L2（单芯片内的一切），PyPTO 的 "L3" 对应 simpler 的 L3
> 及以上（跨芯片的一切）。完整模型见 simpler 的
> [层级化 Level Runtime](https://hw-native-sys.github.io/simpler/hierarchical-level-runtime/)。

分布式章节涵盖 L3。L2 内容见[语言指南](../execution/00-compile.md)。

## 术语表

| 术语 | 定义 |
| ---- | ---- |
| **Rank** | 参与分布式程序的单个进程或芯片。每个 rank 在启动时分配唯一索引。 |
| **Device** | 一个 Ascend NPU 芯片（或 die），由 `device_id` 标识。一个 rank 对应一个 device。 |
| **Node** | 托管一个或多个设备的物理机器。 |
| **Window buffer** | 对称 per-rank HCCL 缓冲区。Rank 通过对等端 `CommContext.windowsIn[peer]`/`windowsOut[peer]` 查看对等端。 |
| **通信域** | 共享对称 window pool 的 rank 子集。默认：整个 world。 |
| **信号** | 跨 rank 同步原语。notify/wait 计数器协调对 window buffer 的访问。 |
| **编排器** | 分配 window buffer 并将 kernel 分发到设备的 HOST 函数。 |
| **InCore kernel** | 在 NPU 上执行的设备端函数。 |

## 阅读路径

1. **[00-model](00-model.md)** — 快速开始优先：运行 2-rank 程序，然后了解模型词汇
2. **[01-collectives](01-collectives.md)** — AllReduce、barrier、broadcast、allgather、reduce_scatter、all-to-all
3. **[02-primitives](02-primitives.md)** — notify/wait、remote_load/remote_store、put/get、CommCtx
4. **[03-execution](03-execution.md)** — DistributedWorker 生命周期、DeviceTensor、多程序、环境变量
5. **[04-debugging](04-debugging.md)** — 常见故障模式和诊断标志

## 相关链接

- [入门指南](../00-getting_started.md) — `ir.compile()`、`CompiledProgram`、`DeviceTensor`、`RunConfig`
- [Simpler 运行时](https://hw-native-sys.github.io/simpler/) — 运行时内部机制（调度器、图构建、tensormap）
