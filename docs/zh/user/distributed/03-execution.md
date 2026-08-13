# 执行

一次性的编译并派发调用足以应付快速测试。生产代码则会将设置成本——
fork chip 进程、组装 kernel——分摊到可复用的 `DistributedWorker` 上的多次
派发中。

## DistributedWorker

通过 `compiled.prepare()` 获得。设置（fork、通信引导、kernel 组装）仅执行一次；
分发可执行多次。

```python
with compiled.prepare() as rt:
    rt(host_x, host_out)
    # ... 更多分发 ...
# rt.close() 在退出时自动执行——释放 buffer 并关闭 worker。
```

`compiled.prepare()` 返回一个 `DistributedWorker`（也可以通过
`DistributedWorker(compiled)` 直接构造，可从 `pypto.runtime` 导入，但
`prepare()` 是文档约定的入口）。

### 方法

| 方法 | 描述 |
| ---- | ---- |
| `compiled.prepare(config=None, *, extra_compiled=(), persistent=False, reset_persistent_windows=None, callbacks=None, sub_worker_overrides=None, startup_timeout_s=None)` | 创建 worker、fork 芯片进程，返回 `DistributedWorker`。作为上下文管理器使用。 |
| `rt(x, y, z)` | 单次分发——转换参数，调用 host_orch。 |
| `rt.run(compiled, x, y, z)` | 多程序分发——选择目标程序。 |
| `rt.submit(compiled, x, y, z)` | 有界异步分发——返回 `DistributedRunHandle`。 |
| `rt.alloc_tensor(shape, dtype, *, init=None)` | 分配 worker 常驻的 `DeviceTensor`。`init` 从 host 拷贝（一次性 H2D）。 |
| `rt.free_tensor(tensor)` | 释放 `DeviceTensor`。 |
| `rt.alloc_stacked_tensor(host_w)` | 沿 dim 0 分片 `host_w`——分片 `i` 上传到卡 `i`。返回 `StackedDeviceTensor`。 |
| `rt.free_stacked_tensor(stacked)` | 释放 `StackedDeviceTensor` 的所有分片。 |
| `rt.copy_stacked_from(stacked, host_out)` | 把每个分片 D2H 读回 `host_out`（共享内存，须在 `prepare()` 前分配）。 |
| `rt.release_inherited_host_tensor_refs()` | fork 后释放运行时在父进程中持有的 host 引用。 |
| `rt.close()` | 释放 buffer，关闭芯片 worker。作为上下文管理器时自动调用。 |

### 值得了解的 `prepare()` 参数

- **`config`**——可选的 `RunConfig`，仅用于按给定的 ring sizing 预热
  runtime arena 缓存，使首次派发跳过约 800ms 的冷构建。它**不会被保留**：
  每次派发仍需自己传入 `config=`。预热只有在*首次*派发的 sizing 与预热
  时使用的一致时才有收益——见 `docs/en/dev/05-runtime-ring-sizing.md` 中
  arena 预热一节。
- **`persistent=True`**——在 worker 整个生命周期内保留 CommDomain
  window，而不是每次派发都分配/释放。与 **`reset_persistent_windows`**
  搭配使用，后者决定保留的 window 是否在两次请求之间清零（正确性与
  开销的权衡）。见 `docs/en/dev/06-persistent-l3.md`。
- **`extra_compiled`**——见下方"在同一个 worker 上运行多个程序"。
- **`startup_timeout_s`**——可选地覆盖 Simpler 对 fork worker 层级报告
  启动就绪状态所设置的正有限秒数期限。保持为 `None` 时使用 Simpler
  默认值；对于确实较慢的冷启动，应增大该期限，而不是取消期限约束。

### 有界异步分发

`DistributedWorker.submit(compiled, *args)` 在 Simpler 接受分发后返回
`DistributedRunHandle`。后端支持异步执行时，调用方可以在当前请求仍在执行时准备
下一请求的 host 工作。`run()` 和 `rt(...)` 仍是阻塞兼容接口。

worker 固定拥有两个可复用的分发元数据帧。前两次提交可以同时处于执行中；第三次
`submit()` 会先等待最老的 handle，再构造和发布新的分发。每个 handle 会快照本次
运行配置，并把参数和生成的任务元数据保留到完成。

使用 `handle.result(timeout)` 或其别名 `handle.wait(timeout)` 等待完成并抛出缓存的
分发错误；`handle.done` 可无阻塞地报告是否已经结束。

```python
with compiled.prepare() as rt:
    first = rt.submit(compiled, input_a, weight, output_a)
    second = rt.submit(compiled, input_b, weight, output_b)
    first.result()
    second.result()
```

重叠分发必须使用不同的可变输入和输出 buffer；对应的 `result()` 返回前不得修改或
释放这些 buffer。只读常驻权重可以共享。关闭 worker 时会按 FIFO 顺序排空所有已接受
的 handle。诊断用双遍 swimlane 采集仍保持同步，并返回一个已经完成的 handle。

## DeviceTensor

设备驻留 buffer，跨分发存活。当 `DeviceTensor` 作为参数传给已编译程序时，
运行时会跳过 H2D/D2H 拷贝——设备上已经有数据。

```python
import torch
from pypto.runtime import DeviceTensor

with compiled.prepare() as rt:
    weight = rt.alloc_tensor((1024, 4096), torch.float16, init=host_weight)
    rt(x, weight, out)   # 通过 worker 分发——无 H2D/D2H
```

## StackedDeviceTensor

跨设备分片——通过 `rt.alloc_stacked_tensor()` 获得：

```python
# Host 张量沿 dim 0 分片——分片[i] 存在卡 i 上。
host_weights = torch.randn(4, 1024, 4096).share_memory_()  # 4 个分片
with compiled.prepare() as rt:
    stacked = rt.alloc_stacked_tensor(host_weights)
    rt(x, stacked, out)
```

> **致命陷阱：** `host_weights` 必须在 `prepare()` **之前**调用
> `.share_memory_()`。上传操作在已经 fork 出的芯片 worker 内部运行，
> 该进程只能读取它在 fork 时继承到的 host 内存——传入普通
> `torch.Tensor` 会在 `alloc_stacked_tensor()` 处抛出 `ValueError`。

## One-Shot vs 持久 Worker

### One-Shot

```python
import torch
from pypto.ir.distributed_compiled_program import DistributedConfig
from pypto.runtime import RunConfig

dc = DistributedConfig(device_ids=[0, 1, 2, 3])
cfg = RunConfig(platform="a2a3", distributed_config=dc)
compiled = orchestrator.compile(config=cfg)   # 从 orchestrator 自身的类型标注读取形状

inputs = torch.randn(4, 1, 256)
outputs = torch.zeros_like(inputs)
compiled(inputs, outputs)   # 阻塞直到所有 rank 完成
```

### 持久 Worker（重复派发）

在多次派发之间复用同一个 worker 对象——这是任何 `DistributedWorker` 的
默认生命周期。（不要与上文的 `persistent=True` CommDomain-window 保留
标志混淆——那是一个可选项，用于跳过每次派发的 window 分配/释放；本节
展示的分摊 fork/通信引导开销，无论 `persistent=` 取值如何都会发生。）

```python
host_x = torch.zeros((4, 1, 256), dtype=torch.float32).share_memory_()
host_out = torch.zeros_like(host_x).share_memory_()

with compiled.prepare() as rt:
    for step in steps:
        host_x.copy_(next_input(step))
        rt(host_x, host_out)
        consume(host_out)
```

> **致命陷阱：** 传入 `DistributedWorker` 的 IO buffer 必须在 `prepare()`
> 前调用 `.share_memory_()`。若忘记，运行时会在分发时拒绝该
> buffer——子进程无法访问父进程的私有内存。

## 在同一个 worker 上运行多个程序

一个 `DistributedWorker` 可以分发多个已编译程序：

```python
compiled_a = ir.compile(ProgramA, platform="a2a3", distributed_config=dc)
compiled_b = ir.compile(ProgramB, platform="a2a3", distributed_config=dc)

with compiled_a.prepare(extra_compiled=[compiled_b]) as rt:
    rt.run(compiled_a, host_x, host_out)  # 分发 ProgramA
    rt.run(compiled_b, host_x, host_out)  # 分发 ProgramB
```

worker 复用其芯片进程和通信设置——没有 fork 开销。`compiled_b` 必须通过
`extra_compiled=` 传入，`rt.run(compiled_b, ...)` 才能找到它；传入未注册的
程序会抛出 `ValueError`。准备多个程序也会让 worker 进入多程序模式，此时
`rt(*args)` 快捷方式含义不明确会抛出 `TypeError`——包括主程序在内的每个
程序都必须显式通过 `rt.run(...)` 派发。

## CLI 启动

分布式程序的启动方式与单设备程序完全一样——见 [00-model](00-model.md)
中的"启动命令"一节：直接 `python script.py`，不需要单独的多进程启动器。

## 环境变量

### 编译时宏

这些是 C 预处理器 `#define` 宏（定义在 `profiling_config.h`），**不是环境
变量**。默认值为 `1`（开启），通过 CMake 编译参数设置；作为 shell 环境
变量设置对其无效。

| 宏 | 默认值 | 效果 |
| -- | ------ | ---- |
| `SIMPLER_HOST_STRACE` | `1`（开） | `benchmark()` 计时标记在编译期所必需。缺失时 `benchmark()` 会抛出 `RuntimeError`。 |
| `SIMPLER_DFX` | `1`（开） | 设备端分析总开关（编排器/调度器指标、PMU 计数器、scope 统计、swimlane trace）。子开关都需要它为 `1`。 |

### 运行时环境变量

| 变量 | 默认值 | 效果 |
| ---- | ------ | ---- |
| `SIMPLER_DEVICE_STRACE_ENABLE` | 开（未设置或非 `"0"`） | 运行时切换设备域 `[STRACE]` 标记。设为 `0` 可在保留 host 标记的同时抑制设备标记。 |

### 基准测试环境变量

`pypto-lib` 的 golden 基准测试框架读取 `PYPTO_BENCH` /
`PYPTO_BENCH_ROUNDS` / `PYPTO_BENCH_WARMUP` / `PYPTO_BENCH_RAW`——这些
在本仓库中未定义也未被使用。其当前默认值见 `pypto-lib` 自身的文档。
`pypto.runtime.benchmark()`（本仓库自己的基准测试工具）在性能指南中
单独说明。

## 相关链接

- [00-model](00-model.md) — 快速开始和模型词汇
- [04-debugging](04-debugging.md) — 常见故障模式
- [入门指南](../00-getting_started.md) — 运行时设置
