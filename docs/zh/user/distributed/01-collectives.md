# 集合通信

本页介绍六种内置集合通信及算法选择。所有集合通信在各 rank 间**同步执行**——
每个 rank 必须以相同形状的 signal tensor 调用同一集合通信，否则程序会挂起
或静默数据损坏。

> **说明：** 以下代码块均为示意性片段——每段仅展示该集合通信相关的原语调用，
> 省略了 `my_rank`/`nranks` 推导、buffer 分配、输入数据搬入等设置代码，并非
> 可直接运行的程序。可运行版本见下方"可运行示例"一节。

## AllReduce

每个 rank 提交其本地数据；每个 rank 接收规约结果（`op=` 选择 `Sum`、`Max`、
`Min` 或 `Prod`）。

```python
# Host 编排器——最简形式（编译器合成 signal）。
data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)  # mesh 模式，就地

# InCore kernel——显式 signal。
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="mesh")
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")

# Host 编排器——把一次调用分摊到每个 rank 的 4 个 AIV 核上。
data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum, core_num=4)
```

### Mesh 模式

- 每步 O(N) 远程流量——每个 rank 读取所有对端
- 每次调用一个全局屏障（AtomicAdd/Ge 在 `[NR, core_num]` signal 上）
- 支持 `pl.dynamic("NR")`
- 最适合小消息和低延迟

### Ring 模式

- 2(P-1) 步：reduce-scatter + allgather
- 每步 O(N/P) 远程流量——每个 rank 读取一个邻居
- Signal 形状：`[2 × (NR − 1), NR]`
- 要求编译时已知 NR——使用工厂函数模式：外层 Python 函数接收 `nr`/`size`，
  推导出 `total_rounds = 2 * (nr - 1)`，并在自己的函数体内定义程序，使
  `[total_rounds, nr]` 成为编译期常量。
- 两种装饰器系列都支持这一模式。`@pl.program` / `@pl.function` 会在装饰时
  捕获*定义处*帧的局部变量；`@pl.jit` 系列则会把闭包常量折叠进它重新生成
  的源码，因此在 HOST 编排体内引用的工厂常量同样能够解析。类形式的写法参见
  下方"可运行示例"一节中的
  `collectives/test_l3_tensor_allreduce_ring_intrinsic.py`。
- 最适合大消息（>16 KiB）和高带宽

| 方面 | Mesh | Ring |
| ---- | ---- | ---- |
| 每步远程流量 | O(N) | O(N/P) |
| 屏障轮次 | 1 | 2(P-1) |
| Signal 形状 | `[NR, core_num]` | `[2 × (NR − 1), NR]` |
| 最适合 | 小消息，低延迟 | 大消息，高带宽 |

**经验法则：** 默认使用 `mode="mesh"`。当负载超过约 16 KiB 且 mesh 带宽达到平台期时
切换到 `mode="ring"`。

Host 编排器形式（省略 `signal`）是语法糖——编译器合成 `[world_size(), core_num]` 的
signal（仅限 mesh）。

### 多核（`core_num`）

在 host 编排器上，`core_num` 把一次 AllReduce 调用分摊到**每个 rank** 的多个
AIV 核上。它不改变任务层级：`device=r` 仍然选择卡；该 rank 的 builtin task 现在
启动一个包含 `core_num` 个 block 的同步 grid，以 block-cyclic 方式把负载切成
256 元素的 tile。

```python
data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum, core_num=4)
```

- 默认为 `1`（单 block，即原有行为）。
- 仅 mesh：`mode="ring"` 要求 `core_num == 1`。
- 不得超过目标平台的 AIV 核数（910B 为 48，950 为 36）——该 launch 要求所有
  block 同时准入，因此超额请求会在编译期被拒绝。
- 显式 `signal` 需要每个 block 一条 lane：`[world_size(), stride]` 且
  `stride >= core_num`。rank-1 的 `[world_size()]` signal 只适用于 `core_num=1`。
- InCore kernel 保持 `core_num=1`，改用外层 `pl.spmd(...)`。

### 变更

`target: InOut` — 数据既被读取（作为规约输入）又被写入（作为规约结果）。所有 rank
必须传入形状相同的 `target` tensor。

### 支持的 ReduceOp

全部四种——`Sum`、`Max`、`Min`、`Prod`——InCore 组合调用和 Host 内置的
mesh 路径均支持。Host 内置的 ring 路径（`builtin.tensor.allreduce_ring`）
更窄：仅 `Sum`，且 target 须为 4 字节的 `FP32`（编译期检查）。mesh 路径的
`target` dtype 必须是 `FP16` 或 `FP32`；ring 路径仅 `FP32`。除了本页开头
要求的形状相同的 signal tensor 外，所有 rank 还必须使用相同的 `ReduceOp`
和 `mode`。

## Barrier

跨 rank 屏障——阻塞直到所有 rank 到达。

```python
# signal: pld.DistributedTensor[[NR, 1], pl.INT32]，新分配的。
signal = pld.tensor.barrier(signal)
```

在 signal 上使用自清理信用屏障（`AtomicAdd(+1)` / `Ge(1)` 并带重置尾声），
因此同一个 signal buffer 可在连续调用间复用。

## Broadcast

将 root rank 的数据广播到所有 rank。

```python
# Root 在调用前写入数据。
if my_rank == ROOT_RANK:
    data = pl.store(local, [0, 0], data)
data = pld.tensor.broadcast(data, signal, root=ROOT_RANK)
# 调用后每个 rank 的 data[0, 0:SIZE] 都持有 root 的数据。
```

Root 必须在调用前写入数据；非 root rank 的输入槽位会被忽略。调用结束后，
每个 rank 都持有 root 的数据。

## AllGather

推式 all-gather——每个 rank 推送自己的本地分片，每个 rank 都收到完整的
汇总矩阵。

```python
# Stage buffer：本 rank 的 [1, SIZE] 分片（推送源）。
stage_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
stage = pld.window(stage_buf, [1, SIZE], dtype=pl.FP32)
stage = pl.store(local_input, [0, 0], stage)

# 结果 buffer：汇总后的 [NR, SIZE]（推送目标）。
data_buf = pld.alloc_window_buffer(NR * SIZE * pl.FP32.get_byte())
data = pld.window(data_buf, [NR, SIZE], dtype=pl.FP32)
sig_buf = pld.alloc_window_buffer(NR * pl.INT32.get_byte())
sig = pld.window(sig_buf, [NR], dtype=pl.INT32)

data = pld.tensor.allgather(stage, data, sig)
# 调用后 data[src, :] 持有 rank src 的分片，对所有 src 均成立。
```

`local_data` 和 `target` **必须是不同的** window buffer。stage buffer 是
每个 rank 的推送源；target buffer 接收汇总后的 `[NR, SIZE]` 结果。

## ReduceScatter

Reduce-scatter：每个 rank 写入全部 NR 个分片，接收自己的规约结果分片。

```python
# 屏障用 signal（host builtin 要求一维）。
sig_buf = pld.alloc_window_buffer(NR * pl.INT32.get_byte())
sig = pld.window(sig_buf, [NR], dtype=pl.INT32)

# 将全部 NR 个分片写入 data[NR, SIZE]。
for j in pl.range(nranks):
    data = pl.store(chunk_j, [j, 0], data)
data = pld.tensor.reduce_scatter(data, sig, op=pld.ReduceOp.Sum)
# data[my_rank, 0:SIZE] 持有本 rank 的规约结果分片。
```

## AllToAll

个性化 all-to-all 交换——每个 rank 向每个对端发送一个专属分片，并从每个
对端接收一个专属分片。

```python
# Stage buffer：推送源，[NR, SIZE]，每行是发往对应目标的分片。
stage_buf = pld.alloc_window_buffer(NR * SIZE * pl.FP32.get_byte())
stage = pld.window(stage_buf, [NR, SIZE], dtype=pl.FP32)
for dest in pl.range(nranks):
    stage = pl.store(chunk_for_dest, [dest, 0], stage)

# 结果 buffer：推送目标，[NR, SIZE]。
data_buf = pld.alloc_window_buffer(NR * SIZE * pl.FP32.get_byte())
data = pld.window(data_buf, [NR, SIZE], dtype=pl.FP32)
sig_buf = pld.alloc_window_buffer(NR * pl.INT32.get_byte())
sig = pld.window(sig_buf, [NR], dtype=pl.INT32)

data = pld.tensor.all_to_all(stage, data, sig)
# data[src, :] 持有从 rank src 收到的分片。
```

`input` 和 `target` 必须是**不同的** window buffer。

## InCore 手写 vs Host 级别内置集合通信

PyPTO 有三种方式运行集合通信——根据代码运行的位置以及是否需要
`mode="ring"` 来选择：

| 方面 | InCore 手写 | InCore 组合调用 | Host 级别内置 |
| ---- | ----------- | --------------- | ------------- |
| **位置** | `@pl.jit.incore` | `@pl.jit.incore` | `@pl.jit.host` |
| **实现** | 手写 `notify`/`wait` + `remote_load` 循环 | 直接调用 `pld.tensor.allreduce(data, sig, ...)` | 直接调用 `pld.tensor.allreduce(data, [sig,] ...)` |
| **Lowering** | 自行实现原语 | `LowerCompositeOps` | `LowerHostTensorCollectives` |
| **支持的模式** | 取决于自己的实现 | `mesh` 和 `ring` | `mesh` 和 `ring`（ring：仅 `Sum` + `FP32`） |
| **Signal 形状** | 取决于自己的分配 | mesh 为 `[nranks, 1]`（rank 数量可为动态）；ring 为 `[2×(NR−1), NR]`（`NR` 必须是编译期常量） | mesh：一维 `[world_size]` 或二维 `[world_size, 1]`（编译器合成的 signal 为二维）；ring：`[2*(NR−1)+1, NR]` |
| **适用场景** | 学习、自定义协议 | ring 需要非 `Sum`/非 `FP32`，或已身处 InCore kernel 内部 | 日常的 host 编排集合通信 |

日常 host 编排代码优先使用 Host 级别内置——它们自动处理屏障编排和分块。
只有 `allreduce` 可以省略 signal 参数（编译器会在循环外自动合成一个）；
其余五种集合通信（`barrier`、`broadcast`、`allgather`、`reduce_scatter`、
`all_to_all`）始终需要调用方显式分配并传入 signal。InCore 组合调用与 Host
内置均支持 `mode="ring"`；当 ring 需要 `Sum` 以外的 `ReduceOp` 或非 `FP32`
的 dtype 时改用 InCore 组合调用，因为 Host 内置的 ring 路径仅支持 `Sum` + `FP32`。

## 可运行示例

上面每种集合通信在 `tests/st/distributed/` 下都有可运行的对应测试
（以下路径均相对该目录）。[教程](05-tutorials.md)页面是面向用户的对应内容——
先手工构建每种集合通信，再揭示内置原语：

| 集合通信 | 教程步骤 | 先手工？ |
| -------- | -------- | -------- |
| barrier | [09-barrier](09-barrier.md) | 是（步骤 04，然后揭示） |
| allreduce | [13-allreduce_mesh](13-allreduce_mesh.md) · [14-allreduce_two_phase](14-allreduce_two_phase.md) · [15-allreduce_ring](15-allreduce_ring.md) · [16-allreduce_reveal](16-allreduce_reveal.md) | 是（步骤 08–11） |
| broadcast | 规划中——步骤 12 | 是 |
| allgather | 规划中——步骤 13 | 是 |
| reduce_scatter | 规划中——步骤 14 | 是 |
| all_to_all | 规划中——步骤 15 | 是 |

| 集合通信 | InCore 手写 | InCore 组合调用 | HOST 内置 |
| -------- | ----------- | --------------- | --------- |
| allreduce | `collectives/test_l3_allreduce.py` | `collectives/test_l3_tensor_allreduce_intrinsic.py` | `test_l3_host_tensor_allreduce.py` |
| allreduce（ring） | `collectives/test_l3_allreduce_ring.py` | `collectives/test_l3_tensor_allreduce_ring_intrinsic.py` | `test_l3_host_tensor_allreduce_ring.py` |
| barrier | — | `collectives/test_l3_tensor_barrier_intrinsic.py` | `test_l3_host_tensor_barrier.py` |
| broadcast | `collectives/test_l3_broadcast.py` | `collectives/test_l3_tensor_broadcast_intrinsic.py` | `test_l3_host_tensor_broadcast.py` |
| allgather | `collectives/test_l3_allgather.py` | `collectives/test_l3_tensor_allgather_intrinsic.py` | `test_l3_host_tensor_allgather.py` |
| reduce_scatter | `collectives/test_l3_reduce_scatter.py` | `collectives/test_l3_tensor_reduce_scatter_intrinsic.py` | `test_l3_host_tensor_reduce_scatter.py` |
| all_to_all | `collectives/test_l3_all_to_all.py` | `collectives/test_l3_tensor_all_to_all_intrinsic.py` | `test_l3_host_tensor_all_to_all.py` |

## 相关链接

- [00-model](00-model.md) — 快速开始和模型词汇
- [02-primitives](02-primitives.md) — 集合通信的底层基础
- [04-debugging](04-debugging.md) — 常见故障模式
