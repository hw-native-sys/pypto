# 分布式编程模型

> **前置知识：** [快速入门](../00-getting_started.md) — 基本 PyPTO tensor/tile 模型。
> 本指南使用 `pld` 命名空间（`import pypto.language.distributed as pld`）。
>
> **DSL 形式：** 本章使用 `@pl.jit`（普通 Python 函数）编写程序。
> `@pl.program`/`@pl.function` 是等价的类形式，用于 `tests/st/distributed/` 下的
> 旧测试——完整 `@pl.jit` 系列见[编译](../execution/00-compile.md)。

## 快速开始：2-Rank AllReduce

最简单的分布式程序——两个 rank 对各自数据求和，结果相同。

> 这是 **mesh all-reduce** 模式——stage in、barrier、读取每个对端的 slice
> 并求和——[教程阶梯](05-tutorials.md)中的
> [mesh allreduce 教程](13-allreduce_mesh.md)逐步构建的正是它。在两个 rank
> 时所有算法都坍缩为同一次交换；[ring allreduce 教程](15-allreduce_ring.md)
> 是揭示内置原语之前的最后一个手工步骤。

```python
import pypto.language as pl
import pypto.language.distributed as pld

NR = pl.dynamic("NR")
SIZE = 256

@pl.jit.incore
def reduce_step(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
    signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
) -> pl.Tensor[[1, SIZE], pl.FP32]:
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)

    # 1. Stage-in：将本地输入数据复制到本 rank 的 window 分片。
    local = pl.load(inp, [0, 0], [1, SIZE])
    data = pl.store(local, [0, 0], data)

    # 2. Barrier：通知每个对端，然后等待每个对端。
    for peer in pl.range(nranks):
        if peer != my_rank:
            pld.system.notify(
                signal, peer=peer, offsets=[my_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )
    for src in pl.range(nranks):
        if src != my_rank:
            pld.system.wait(
                signal, offsets=[src, 0],
                expected=1, cmp=pld.WaitCmp.Ge,
            )

    # 3. 计算：累加每个对端的分片。
    acc = pl.load(data, [0, 0], [1, SIZE])
    for peer in pl.range(nranks):
        if peer != my_rank:
            peer_tile = pld.tile.remote_load(
                data, peer=peer, offsets=[0, 0], shape=[1, SIZE]
            )
            acc = pl.add(acc, peer_tile)

    # 4. Stage-out：将累加器写入本地输出。
    return pl.store(acc, [0, 0], out)

@pl.jit
def chip_orch(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
    signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
) -> pl.Tensor[[1, SIZE], pl.FP32]:
    # 逐设备的编排包装函数——HOST 派发的是它，而不是直接派发 InCore kernel。
    return reduce_step(inp, out, data, signal)

@pl.jit.host
def orchestrator(
    inputs: pl.Tensor[[NR, 1, SIZE], pl.FP32],
    outputs: pl.Out[pl.Tensor[[NR, 1, SIZE], pl.FP32]],
) -> pl.Tensor[[NR, 1, SIZE], pl.FP32]:
    data_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
    signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())

    for r in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
        chip_orch(inputs[r], outputs[r], data, signal, device=r)
    return outputs
```

### 运行程序

将上面的函数和下面的驱动代码保存到同一个文件（`script.py`）中：

```python
import torch
from pypto.runtime import RunConfig
from pypto.ir.distributed_compiled_program import DistributedConfig

dc = DistributedConfig(device_ids=[0, 1])
cfg = RunConfig(platform="a2a3", distributed_config=dc)

inputs = torch.randn(2, 1, SIZE)
outputs = torch.zeros_like(inputs)
orchestrator(inputs, outputs, config=cfg)   # 阻塞直到两个 rank 都完成
```

这是"one-shot"派发模式。`DistributedWorker`、持久 worker 和多程序派发见
[03-execution](03-execution.md)。

### 预期输出

每个 rank `r` 的 `outputs[r] == sum(inputs[*])`。2 个 rank 输入分别为
`[[1, 2, 3]]` 和 `[[10, 20, 30]]`，两个 rank 均看到 `[[11, 22, 33]]`。

### 启动命令

```bash
python script.py
```

不涉及任何多进程启动器——运行时会从这一个 Python 进程中为每个设备（按
`device_ids`）fork 出一个 worker 进程。

## PyPTO 中的分布式编程是什么？

PyPTO 的分布式模型采用**对称内存 + 信号**范式。每个 rank 有一个 per-rank
**window buffer**，各对端的地址空间是对称的。通信通过单边
`put`/`get`/`remote_load` 加**信号同步**（`notify`/`wait`）来完成。
**通信域** 是共享对称 window pool 的 rank 子集；整个 world 为默认通信域。

编译器 lowering 出的每个 allreduce、broadcast、barrier 都是这些相同原语的
组合——`pld.tensor.*` 集合通信（`allreduce`、`barrier` 等）是它们的语法糖，
而非另一套独立的库。完整 API 见 [01-collectives](01-collectives.md)。

## 模型

### HOST 编排器

HOST 函数分配 window buffer、分发 kernel 并管理控制平面：

- 声明为 `@pl.jit.host`
- 调用 `alloc_window_buffer`、`window()`，通过 `device=r` 进行 per-rank 分发
- 每个进程运行一次——不在 NPU 上执行
- 派发的是逐设备的 `@pl.jit` 包装函数（而非直接派发 `InCore` kernel）——见下方
  Per-Rank 分发

### InCore Kernel

InCore 函数在 NPU 设备上运行：

- 声明为 `@pl.jit.incore`
- 接收 window-bound 的 `DistributedTensor` 参数
- 使用 `notify`/`wait` 进行跨 rank 同步，`remote_load`/`remote_store` 进行 RMA
- 不调用 `alloc_window_buffer` 或 `world_size()`

### Per-Rank 分发

HOST 从不直接派发 `InCore` 函数。它通过设置 `device=r` 派发一个逐设备的
`@pl.jit` 包装函数；该包装函数再调用 `InCore` kernel，且不带 `device=` 参数。
每个 rank 通过 `CommContext` 看到
自己的对称 window buffer 视图。

### Window Buffer 生命周期

`alloc_window_buffer(size)` 创建 per-rank buffer。`window(buf, shape, dtype)`
创建类型化视图。Buffer 在 host 编排器调用期间存活；**默认情况下**编排器
调用之间没有持久化的 IPC——如需跨多次派发保留 window，见
[03-execution](03-execution.md) 中的 `persistent=True` 选项。

### 控制平面 vs 执行平面

```text
HOST 编排器 (@pl.jit.host)
  ├── alloc_window_buffer(...)   ← 控制平面：声明布局
  ├── window(buf, shape, dtype)  ← 控制平面：创建类型化视图
  └── for r in ranks:            ← 分发循环
        chip_orch(..., device=r) ← 桥接到逐设备包装函数

编排包装函数 (@pl.jit)
  └── reduce_step(...)           ← 调用 InCore kernel，不带 device=

InCore kernel (@pl.jit.incore)
  ├── notify / wait               ← 执行平面：跨 rank 同步
  ├── remote_load                 ← 执行平面：读取对端数据
  └── store                       ← 执行平面：写入本地输出
```

## 逐行解读

| 代码 | 说明 |
| ---- | ---- |
| `NR = pl.dynamic("NR")` | world size 在构建时未知。`pl.dynamic` 将维度推迟到运行时分发——host 从 `len(device_ids)` 绑定。 |
| `pl.InOut[pld.DistributedTensor[...]]` | `data` 和 `signal` 是 window-bound：每个 rank 共享相同的地址空间布局。`InOut` 表示 kernel 既读又写。 |
| `pld.get_comm_ctx(data)` | 将 window-bound tensor 提升为通信域句柄。每个 rank 获得自己的 `ctx`，从中读取 `rank()` 和 `nranks()`。 |
| `pld.system.notify(..., op=AtomicAdd)` | 每个 rank 原子地加 1 到每个对端的 signal slot。使用 `offsets=[my_rank, 0]` 时，每个格子只有一个写者，因此这里用 `Set` 效果完全相同——展示 `AtomicAdd` 是因为本文档中每个 barrier 用的都是同一个 notify 调用。`AtomicAdd` 仅在*共享*格子的 barrier（多个 rank 写入同一个 slot）中才是必需的；区别见 02-primitives.md 中的"隔离的 Barrier"一节。 |
| `pld.system.wait(..., cmp=Ge, expected=1)` | 阻塞直到本地 signal slot 达到至少 1——表示所有对端的 notify 均已到达。 |
| `pld.tile.remote_load(...)` | 读取对端 `DistributedTensor` 的远程分片到本地 tile。这是 `pl.tile.load` 的 tile 级别跨 rank 版本。 |
| `pl.add(acc, peer_tile)` | 本地加法循环累加所有对端贡献。循环结束后 `acc` 持有 `sum(inputs[*])`。 |
| `chip_orch`（`@pl.jit`） | HOST 通过 `device=r` 派发这个逐设备包装函数，而不是直接派发 `InCore` kernel；它再以不带 `device=` 的方式调用 `reduce_step`。 |
| `inputs[r]` / `outputs[r]` | 下标索引会去掉最前面的 rank 维度，得到 `reduce_step` 声明的二维 `[1, SIZE]` 形状——若改用带显式 shape 的 `pl.slice`，会保留该维度，导致形状不匹配。 |

## 相关链接

- [05-tutorials](05-tutorials.md) — 逐步的分布式教程阶梯
- [01-collectives](01-collectives.md) — 内置集合通信及其语义
- [02-primitives](02-primitives.md) — 集合通信的底层基础
- [03-execution](03-execution.md) — DistributedWorker 生命周期和生产模式
- [04-debugging](04-debugging.md) — 常见故障模式和诊断标志
