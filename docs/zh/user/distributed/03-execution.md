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
| `rt.copy_to(dst_dev_ptr, src_host_ptr, nbytes, *, worker_id=0)` | 显式 staged H2D 拷贝。host `torch.Tensor` 源只需为 CPU 连续张量，可在 `prepare()` 后创建。 |
| `rt.copy_from(dst_host_ptr, src_dev_ptr, nbytes, *, worker_id=0)` | 显式 staged D2H 拷贝。host `torch.Tensor` 目标只需为 CPU 连续张量，可在 `prepare()` 后创建。 |
| `rt.alloc_stacked_tensor(host_w)` | 沿 dim 0 分片 `host_w`——分片 `i` 上传到卡 `i`。返回 `StackedDeviceTensor`。 |
| `rt.free_stacked_tensor(stacked)` | 释放 `StackedDeviceTensor` 的所有分片。 |
| `rt.copy_stacked_from(stacked, host_out)` | staged D2H 读回 CPU 连续的 `host_out`；可在 `prepare()` 后分配。 |
| `rt.committed_device_memory(worker_id=0)` | 本 worker 自己的分配器在卡 `worker_id` 上已提交的设备 HBM 字节数——张量、池化 arena、运行时 buffer。多卡总量需按 id 求和。 |
| `rt.device_memory_info(worker_id=0)` | `worker_id` 所在整张卡的 `(free_bytes, total_bytes)`，即驱动看到的视图。用它来决定分配多大；卡上其他任何东西都会改变 `free_bytes` 而不改变已提交总量。模拟器后端会抛异常，而不是报告 0。 |
| `rt.release_inherited_host_tensor_refs()` | 释放父进程中为兼容保留的生命周期引用。 |
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

### 常驻张量的所有权

常驻参数只能用于 prepared worker。`DeviceTensor` 必须由执行它的同一个
`DistributedWorker.alloc_tensor` 返回，`StackedDeviceTensor` 必须由该 worker
的 `alloc_stacked_tensor` 返回。这些分配接口会在每个设备张量或分片上保留 Simpler
owner `Buffer`，使无地址 wire ABI 能构造有效的 Tensor descriptor。手工包装裸指针，
或把常驻张量交给另一个 worker，都会被拒绝。

## DeviceTensor

设备驻留 buffer，跨分发存活。当 `DeviceTensor` 作为参数传给已编译程序时，
运行时会跳过 H2D/D2H 拷贝——设备上已经有数据。

```python
import torch

with compiled.prepare() as rt:
    weight = rt.alloc_tensor((1024, 4096), torch.float16, init=host_weight)
    rt(x, weight, out)   # 通过 worker 分发——无 H2D/D2H
```

## StackedDeviceTensor

跨设备分片——通过 `rt.alloc_stacked_tensor()` 获得：

```python
# Host 张量沿 dim 0 分片——分片[i] 存在卡 i 上。
with compiled.prepare() as rt:
    host_weights = torch.randn(4, 1024, 4096).contiguous()  # prepare() 后的 host 张量
    stacked = rt.alloc_stacked_tensor(host_weights)
    rt(x, stacked, out)
```

通过 `rt.alloc_tensor(init=...)`、`rt.alloc_stacked_tensor(...)`、
`rt.copy_to(...)`、`rt.copy_from(...)` 和 `rt.copy_stacked_from(...)` 执行的
显式常驻上传与读回，都会经过 runtime 管理的 POSIX 共享内存 staging。host 端为
`torch.Tensor` 时只需是 CPU 连续张量，可以是在 `prepare()` 后创建的普通张量；
无需 `.share_memory_()`、fork 前分配或 `inherited_host_tensors`。

### 跳过 staging 拷贝

staging 需要在 host 侧完整拷贝一份数据；对体积很大的常驻权重，这份拷贝值得省掉。
通过 `inherited_host_tensors` 注册的区间会被直接按地址命名，既不需要 staging buffer
也不需要 memcpy：它在 fork 之前就已存在，每个子进程都在同一地址看到它。

**列入该列表即是你作出的保证。** 把一个张量传入 `inherited_host_tensors`，等于断言两点：

- 它的后备内存**跨进程可见**——即 `MAP_SHARED` 映射，无论是 torch 自己的共享内存
  还是外部文件映射；并且
- 该映射在 worker 的整个生命周期内**始终有效**。

传入 `MAP_PRIVATE` 后备内存属于**不受支持**的用法。写时复制会让子进程一直读到 fork
之前的快照，因此上传的数据可能是陈旧或错误的。这一点不会被自动检测出来。

PyPTO 无法验证这项保证，也不去尝试。`is_shared()` 回答的是另一个问题——storage 是否
为 torch 的共享内存分配；而用 `mmap` + `numpy.frombuffer` + `torch.from_numpy` 包装的
只读 `MAP_SHARED` 文件映射确实是共享的，`is_shared()` 却返回 `False`。读取
`/proc/self/maps` 能给出正确答案，但仅限 Linux，而模拟器同样运行在 macOS 上。因此
`is_shared()` 被保留为单向信号：`True` 可确认是 torch 管理的共享内存，`False` 则无从
判断；对于无法确认的张量，PyPTO 会在 `prepare()` 时发出一次 `RuntimeWarning` 后继续
执行。它既不拒绝该张量，也不退回 staging——退回 staging 会悄悄把你要求省掉的那份拷贝
重新加回来。

```python
weights = map_readonly_shared(path)          # mmap 支撑的 MAP_SHARED，is_shared() == False
with compiled.prepare(
    inherited_host_tensors=[weights],        # “每个子进程都可见，且在我的生命周期内有效”
) as rt:
    resident = rt.alloc_stacked_tensor(weights)
```

这项保证针对的是可见性，因此两个方向都成立：被列入的区间既可作上传源，也可作读回目标。
可写性交由硬件负责——以只读文件描述符建立的 `MAP_SHARED` 映射在写入时会直接触发错误，
而不会破坏任何数据。未列入的张量，或在 `prepare()` 之后分配的张量，其行为与此前完全一致。

被命名的上传源只授予 `READ`，只有目标才授予 `READWRITE`。这对只读映射很关键：把它
描述为可写，等于告诉 runtime 消费者可以写一段写入即出错的内存。

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

one-shot 只接受 host `torch.Tensor` 参数。它会拒绝 `DeviceTensor` 和
`StackedDeviceTensor`；这两种常驻参数都必须使用 prepared worker。

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

> **致命陷阱：** 直接传给 `rt(...)` 或 `rt.run(...)` 的 host `torch.Tensor`
> 参数必须在 `prepare()` 前调用 `.share_memory_()`。若忘记，运行时会在分发时
> 拒绝该 buffer——子进程无法访问父进程的私有内存。此规则不适用于上面列出的
> 显式 staged 上传/读回接口。

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

## 配套示例

`examples/runtime/distributed_callback.py` —— 函数体写成 `...` 的 HOST 级 `SubWorker`，于是它作为
纯 Python 回调运行在 fork 出来的编排进程里。当逻辑无法在编译期写出来时就用这个形态：需要读取实时模型
状态的采样闭包、host 侧指标收集器、结果检查器。

## 相关链接

- [00-model](00-model.md) — 快速开始和模型词汇
- [04-debugging](04-debugging.md) — 常见故障模式
- [入门指南](../00-getting_started.md) — 运行时设置
