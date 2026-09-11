# 分布式编程

PyPTO 的分布式模型建立在**对称内存与信号**之上——完整说明见
[00-model](00-model.md)。简而言之：每个 rank 的 window buffer 与所有对端
具有相同的布局，通过单边 `put`/`get`/`remote_load` 访问其他 rank，并通过
**信号同步**（`notify`/`wait`）进行协调。

编译器 lowering 出的每个 allreduce、broadcast、barrier 都是这些相同原语的
组合——`pld.tensor.*` 集合通信（`allreduce`、`barrier` 等）是它们的语法糖，
而非另一套独立的库。

## 一图看懂对称内存

```text
                    通信域（默认：整个 world）
   ┌───────────────────────┬───────────────────────┬───────────────────────┐
   │        rank 0         │        rank 1         │        rank 2         │
   │  window base: 0x1000  │  window base: 0x5000  │  window base: 0x9000  │
   └───────────────────────┴───────────────────────┴───────────────────────┘
       ▲ 每个 rank 自己的 base 可以不同。每个 rank 还保存一张查找表
         windowsIn[peer]，把每个对端的 rank 索引映射到该对端的 base——
         因此 rank 1 把 rank 0 的数据访问为 windowsIn[0] + offset，而
         不是共享的绝对指针。"对称"指的是每个 window 有相同的大小和布局
         （偏移量 X 在每个 rank 上都是同一块 slice），而不是所有 rank
         共享同一个地址。
```

每个 rank 的 window 具有相同的布局——这才是"对称"的实质——但每个 rank
自己的 base 地址可以不同。每个 rank 都保存一张查找表
（`CommContext.windowsIn[peer]`），记录每个对端的 base，因此
`remote_load`/`remote_store`/`put`/`get` 在本地就能算出
`windowsIn[peer] + offset`，无需向对端询问地址。信号（`notify`/`wait`）是
另一套独立机制，用于告知某个 rank 数据*何时*就绪——仅凭布局本身无法保证
这一点。

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

分布式章节涵盖 L3。L2 内容见[编译](../execution/00-compile.md)。

## 术语表

| 术语 | 定义 |
| ---- | ---- |
| **Rank** | 参与分布式程序的单个进程或芯片。每个 rank 在启动时分配唯一索引。 |
| **Device** | 一个 Ascend NPU 芯片（或 die），由 `device_id` 标识。一个 rank 对应一个 device。 |
| **Node** | 托管一个或多个设备的物理机器。 |
| **对称内存** | 在同一个通信域内，每个 rank 的 window buffer 具有相同的大小与布局——每个 rank 通过自己的 `windowsIn[peer]` 查找表加本地偏移量来访问对端数据，而非共享的绝对地址。参见上方[一图看懂对称内存](#一图看懂对称内存)。 |
| **Window buffer** | 对称 per-rank HCCL 缓冲区。Rank 通过对等端 `CommContext.windowsIn[peer]`/`windowsOut[peer]` 查看对等端。 |
| **Window buffer 地址空间** | 一个 rank 自身 window buffer 所占据的地址范围。在同一通信域内，每个 rank 的 window 有相同的大小与布局——而非相同的绝对地址——这才是该缓冲区"对称"的原因。 |
| **通信域** | 共享对称 window pool 的 rank 子集。默认：整个 world。 |
| **信号** | 跨 rank 同步原语。notify/wait 计数器协调对 window buffer 的访问。 |
| **编排器** | 分配 window buffer 并将 kernel 分发到设备的 HOST 函数。 |
| **InCore kernel** | 在 NPU 上执行的设备端函数。 |

## 阅读路径

1. **[00-model](00-model.md)** — 快速开始优先：运行 2-rank 程序，然后了解模型词汇
2. **[01-collectives](01-collectives.md)** — AllReduce、barrier、broadcast、allgather、reduce_scatter、all-to-all
3. **[02-primitives](02-primitives.md)** — notify/wait、remote_load/remote_store、put/get、CommCtx
4. **[03-execution](03-execution.md)** — DistributedWorker 生命周期、DeviceTensor、多程序、环境变量
5. **[04-debugging](04-debugging.md)** — 常见故障模式、诊断标志，以及分步陷阱索引
6. **[05-tutorials](05-tutorials.md)** — 16 步可运行教程阶梯（`examples/distributed/`）；教程页 `NN` 对应示例 `(NN-5)_*.py`（如 `06-hello_rank.md` 对应 `01_hello_rank.py`）

## 相关链接

- [入门指南](../00-getting_started.md) — `ir.compile()`、`CompiledProgram`、`DeviceTensor`、`RunConfig`
- [Simpler 运行时](https://hw-native-sys.github.io/simpler/) — 运行时内部机制（调度器、图构建、tensormap）
