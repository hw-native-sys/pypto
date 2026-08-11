# 分布式教程（Distributed Tutorials）

`pld` 词汇表按步骤讲解：一个十六步的教程系列，每步一个概念。十一个可运行的
示例现已交付——从 "hello rank" 到点对点移动、动态 rank 数量，以及 all-reduce
三连与其揭示；步骤 12–15（其余集合通信）与步骤 16（组合）为规划中。

> **前置条件：** 先通读[分布式编程](../distributed/index.md)章节——了解词汇，
> 再回到这里亲手构建同样的概念。硬件：步骤 01–06 需要两个设备，步骤 07
> 需要任意 ≥ 2 的数量（三个或更多才能看到环与 P=2 的差异），步骤 08–11
> 的集合通信对比需要四个设备。

## 思路（The idea）

参考章节告诉你 `pld` *是什么*；本教程系列展示它 *做什么*。每个已交付步骤
都是一个小型、经 golden 校验的程序，只教授一个抽象，并且顺序经过设计：
先用原语手工构建每个概念，再让内置原语取代它：

- 步骤 01–02 建立执行模型（rank 身份、三层模型）。
- 步骤 03 引入 **window memory**——所有其他内容接触的基座。
- 步骤 04 在揭示内置原语之前，先用 `notify`/`wait` **手工构建 barrier**。
- 步骤 05–06 覆盖**点对点**移动（`remote_load`/`remote_store`、`put`/`get`）。
- 步骤 07 让 rank 数量变成**动态**（`pl.dynamic("NR")`）：同一份源码可为
  任意 P 编译——这是 P=4 集合通信所依赖的机制。
- 步骤 08–11 用三种方式构建 **all-reduce**（mesh、two-phase、ring），然后
  揭示 `pld.tensor.allreduce`。
- 步骤 12–15 覆盖其余集合通信；步骤 16 组合其中三种。

> **揭示纪律（Reveal discipline）：** 教程页面在揭示它们的步骤之前，不会引入
> 内置原语（`pld.tensor.barrier`、`pld.tensor.allreduce` 等）——本索引仅预告
> 即将出现的内容。等到内置原语出现时，你已经写出手工版本，并知道它们 lower
> 成什么。
>
> **知识递进（Progression）：** 每个步骤只使用更早步骤（或前置章节）引入的
> 概念。当某个步骤提到较晚才讲解的内容——例如步骤 04 的 barrier 揭示中
> 用了一行 `remote_load`——那只是指引，不是必需知识：你可以在后续步骤中
> 正式认识它。

## 建议阅读顺序（Suggested reading order）

按顺序阅读这些步骤——**01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 →
11 → 12 → 13 → 14 → 15 → 16**。每个页面都会重复此顺序块。步骤 01–11 现已交付；
12–16 仍为规划中。

## 16 个步骤

| 步骤 | 程序 | 教授内容 | 状态 |
| ---- | ---- | -------- | ---- |
| 01 | `01_hello_rank.py` | Rank 身份、`pld.world_size()`、`DistributedConfig`；一次 per-rank 分发 | ✅ 已交付 |
| 02 | `02_programming_model.py` | 三层模型：`@pl.jit.host` → `@pl.jit` → `@pl.jit.incore` | ✅ 已交付 |
| 03 | `03_window_buffer.py` | Window memory：`alloc_window_buffer`/`window`；仅自身 slice，无通信 | ✅ 已交付 |
| 04 | `04_barrier.py` | 仅信号：`notify(AtomicAdd)`/`wait(Ge)`；单次汇合 N-rank barrier；揭示 `pld.tensor.barrier` | ✅ 已交付 |
| 05 | `05_remote_load_store.py` | Tile 级 RMA：`remote_load`/`remote_store`；一步环形移位 | ✅ 已交付 |
| 06 | `06_put_get.py` | Tensor 级 p2p：`put`/`get`；push 与 pull | ✅ 已交付 |
| 07 | `07_dynamic_rank_count.py` | 动态 rank 数量：`pl.dynamic("NR")`；同一份源码，任意 P | ✅ 已交付 |
| 08 | `08_allreduce_mesh.py` | All-reduce v1（mesh）：每个 rank 读取所有对端，本地求和 | ✅ 已交付 |
| 09 | `09_allreduce_two_phase.py` | All-reduce v2：reduce-scatter + all-gather | ✅ 已交付 |
| 10 | `10_allreduce_ring.py` | All-reduce v3（ring）：沿环分块 | ✅ 已交付 |
| 11 | `11_allreduce_reveal.py` | **揭示**：`pld.tensor.allreduce`（mesh + ring）；对比 IR | ✅ 已交付 |
| 12 | `12_broadcast.py` | 一对多；揭示 `pld.tensor.broadcast` | 规划中 |
| 13 | `13_allgather.py` | 全对全切片；揭示 `pld.tensor.allgather` | 规划中 |
| 14 | `14_reduce_scatter.py` | 全对分块；揭示 `pld.tensor.reduce_scatter` | 规划中 |
| 15 | `15_all_to_all.py` | 个性化交换；揭示 `pld.tensor.all_to_all` | 规划中 |
| 16 | `16_putting_it_together.py` | 在一个 kernel 中组合 `broadcast` + `allreduce` + `allgather` | 规划中 |

步骤 12–16 为**规划中**——它们将在后续 PR 中交付。下面的教程页面（06–16）
覆盖步骤 01–11。

## 抽象总览（The abstractions map）

每个 `pld` 抽象：一行用途、文档它的章节小节、教授它的教程步骤。
教程的**覆盖契约**：代码中存在的任何内容都必须能由某个示例教授。

### 系统基座（System substrate）

| 抽象 | 用途 | 章节小节 | 运行位置 | 教程步骤 |
| ---- | ---- | -------- | -------- | -------- |
| `pld.world_size()` | world 中的 rank 数量 | [02-primitives](02-primitives.md) §系统基座 | Host（编排器） | 01 |
| `pld.get_comm_ctx(dt)` | 解析 `DistributedTensor` 所属的通信上下文 | [02-primitives](02-primitives.md) §系统基座 | Host / InCore | 04 |
| `pld.rank(ctx)` | 本 rank 在上下文中的索引 | [02-primitives](02-primitives.md) §系统基座 | InCore | 04 |
| `pld.nranks(ctx)` | 上下文中的 rank 数量 | [02-primitives](02-primitives.md) §系统基座 | InCore | 04 |
| `pl.dynamic("NR")` | 命名一个运行期解析的维度（如 rank 数量） | [00-getting_started](../00-getting_started.md) | — | 07 |

### 内存（Memory）

| 抽象 | 用途 | 章节小节 | 教程步骤 |
| ---- | ---- | -------- | -------- |
| `pld.DistributedTensor` | 绑定 window 的张量类型，对端可见 | [00-model](00-model.md) §术语表 | 03 |
| `pld.alloc_window_buffer(...)` | 分配对称 per-rank window buffer | [02-primitives](02-primitives.md) §Window Buffer 管理 | 03 |
| `pld.window(...)` | window buffer 的 `DistributedTensor` 视图 | [02-primitives](02-primitives.md) §Window Buffer 管理 | 03 |

### 信号（Signals）

| 抽象 | 用途 | 章节小节 | 运行位置 | 教程步骤 |
| ---- | ---- | -------- | -------- | -------- |
| `pld.system.notify(...)` | 在对端增加一个信号单元 | [02-primitives](02-primitives.md) §Notify 与 Wait | InCore | 04 |
| `pld.system.wait(...)` | 阻塞直到信号单元达到阈值 | [02-primitives](02-primitives.md) §Notify 与 Wait | InCore | 04 |
| `pld.NotifyOp.AtomicAdd` | 累加的 notify 模式（多写入者安全） | [02-primitives](02-primitives.md) §选择 NotifyOp 与 WaitCmp | — | 04 |
| `pld.WaitCmp.Ge` | wait 模式：`>= expected` 时通过 | [02-primitives](02-primitives.md) §选择 NotifyOp 与 WaitCmp | — | 04 |

### Tile 级 RMA

| 抽象 | 用途 | 章节小节 | 教程步骤 |
| ---- | ---- | -------- | -------- |
| `pld.tile.remote_load(...)` | 将对端的 window slice 拉入本地 tile | [02-primitives](02-primitives.md) §Tile 级 RMA | 05 |
| `pld.tile.remote_store(...)` | 将本地 tile 推入对端的 window slice | [02-primitives](02-primitives.md) §Tile 级 RMA | 05 |

### Tensor 级点对点

| 抽象 | 用途 | 章节小节 | 教程步骤 |
| ---- | ---- | -------- | -------- |
| `pld.tensor.put(...)` | 将本地 window slice 推入对端的 window | [02-primitives](02-primitives.md) §Put 与 Get | 06 |
| `pld.tensor.get(...)` | 将对端的 window slice 拉入本地内存 | [02-primitives](02-primitives.md) §Put 与 Get | 06 |
| `pld.AtomicType` | put/get 的原子性模式 | [02-primitives](02-primitives.md) §Put 与 Get | 06 |

### 集合通信（Collectives）

| 抽象 | 用途 | 章节小节 | 教程步骤 |
| ---- | ---- | -------- | -------- |
| `pld.tensor.barrier(...)` | 同步所有 rank（揭示的内置原语） | [01-collectives](01-collectives.md) §Barrier | 04 |
| `pld.tensor.allreduce(...)` | 归约并广播结果（mesh/ring） | [01-collectives](01-collectives.md) §AllReduce | 11 |
| `pld.tensor.broadcast(...)` | 一个 rank 的数据到全部 | [01-collectives](01-collectives.md) §Broadcast | 12 |
| `pld.tensor.allgather(...)` | 所有 rank 的切片到全部 | [01-collectives](01-collectives.md) §AllGather | 13 |
| `pld.tensor.reduce_scatter(...)` | 归约结果，每个 rank 一个分块 | [01-collectives](01-collectives.md) §ReduceScatter | 14 |
| `pld.tensor.all_to_all(...)` | 个性化交换 | [01-collectives](01-collectives.md) §AllToAll | 15 |

### 组合（Composition）

| 抽象 | 用途 | 章节小节 | 教程步骤 |
| ---- | ---- | -------- | -------- |
| `@pl.jit.host` | 主机编排器：分配 window、分发 rank | [00-model](00-model.md) §术语表 | 02 |
| `@pl.jit` / `@pl.jit.incore` | 每设备编排 / 设备端 kernel | [03-execution](03-execution.md) | 02 |
| `device=r` | 从主机循环将一次分发固定到某个设备 | [00-model](00-model.md) | 01 |
| `DistributedConfig` | 编译的设备列表与 worker 数量 | [03-execution](03-execution.md) | 01 |

## 参见（See also）

- [00-model](00-model.md) — 快速入门与模型词汇
- [01-collectives](01-collectives.md) — 集合通信（步骤 08–16）
- [02-primitives](02-primitives.md) — 集合通信之下的基座
- 下一步：[06-hello_rank](06-hello_rank.md) — 运行你的第一个双 rank 程序
