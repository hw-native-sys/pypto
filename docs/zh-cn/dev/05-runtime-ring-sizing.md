# 单任务 Ring 尺寸配置（Per-Task Ring Sizing）

PyPTO 通过 [`RunConfig`](../../../python/pypto/runtime/runner.py) 上的三个可选
覆盖项暴露 Simpler 的单任务 ring 尺寸配置。它们让你在单次派发中为运行时的
单任务 ring 资源设置尺寸，而无需改动已编译产物或任何全局状态。该能力同时适用于
L2 单 chip 路径（`run()` / `ChipWorker.run()`）和 L3 分布式路径
（`DistributedWorker.run()` / 一次性的 `compiled(...)`）。

运行时把任务启动相关资源保存在 *ring buffer*（环形缓冲）中。每个覆盖项与
Simpler 的 `CallConfig.runtime_env` 上的同名字段一一对应，并且按 **每次任务提交**
生效 —— 即每次 `run()` / `rt.run()` 调用，因此同一 kernel 的不同提交可以使用不同
的 ring 尺寸。在 L3 路径上，覆盖项按每次派发生效，叠加在程序的 `DistributedConfig`
基线（`block_dim` / `aicpu_thread_num`）之上，并作用于该次派发的所有 chip。

## 字段对照表

| `RunConfig` 字段 | `CallConfig.runtime_env` 成员 | 作用 | 约束 |
| ---------------- | ----------------------------- | ---- | ---- |
| `ring_task_window: int \| None` | `ring_task_window` | task ring 中在途 task slot 的数量 | 2 的幂，`>= 4` |
| `ring_heap: int \| None` | `ring_heap` | 每个 ring 的 task 输出堆字节数 | 2 的幂，`>= 1024` |
| `ring_dep_pool: int \| None` | `ring_dep_pool` | 依赖边池容量 | `[4, INT32_MAX]` |

`None`（默认值）表示该字段 **未设置**（在 `CallConfig` 上为 `0`）。PyPTO 仅在值
不为 `None` 时才写入 `CallConfig.runtime_env`，因此未设置的字段完全交由运行时
决定。

## 优先级

对每个值，运行时按如下顺序解析出生效尺寸：

```text
单任务 CallConfig.runtime_env 值   （RunConfig 覆盖 —— 最高优先级）
  └─ 回退到 → PTO2_RING_* 环境变量  （进程级）
       └─ 回退到 → 编译期默认值      （最低优先级）
```

因此 `RunConfig` 覆盖优先于 `PTO2_RING_TASK_WINDOW` / `PTO2_RING_HEAP` /
`PTO2_RING_DEP_POOL` 环境变量，而后者又优先于运行时内置的默认值。

## 校验

`RunConfig` 在构造时（`__post_init__` 中）校验这些覆盖项，与运行时的
`RuntimeEnv::validate()` 保持一致。这样可以在调用处直接抛出清晰的 `ValueError`，
而不是在派发时深陷运行时自身的 `CallConfig::validate()` 失败：

```python
RunConfig(platform="a2a3", ring_heap=1000)
# ValueError: ring_heap must be a power of 2 >= 1024 (bytes per ring), got 1000
```

非整数（包括 `bool`）会以相同的 `ValueError` 被拒绝，而非从 2 的幂判断中抛出
难以理解的 `TypeError`。

## 用法

### L2 单 chip（`run`）

```python
from pypto.runtime import run, RunConfig

compiled = run(
    MyProgram,
    a, b, c,
    config=RunConfig(
        platform="a2a3",
        ring_task_window=128,        # 128 个在途 task slot
        ring_heap=8 * 1024 * 1024,   # 每个 ring 8 MiB 输出堆
        ring_dep_pool=256,           # 256 个依赖边条目
    ),
)
```

### L3 分布式（`DistributedWorker` / `compiled(...)`）

把同样的 `RunConfig` 传给每次派发调用。已准备好的 worker 的共享配置不会被改动，
因此连续多次派发 —— 以及多程序 serving 场景下的不同程序（如 prefill 与 decode）——
都可以各自使用不同的 ring 尺寸：

```python
from pypto.runtime import RunConfig

with compiled.prepare() as rt:           # 仅一次性 setup
    prefill_cfg = RunConfig(platform="a2a3", ring_task_window=256)
    decode_cfg = RunConfig(platform="a2a3", ring_task_window=64)
    rt(host_x, weight, host_out, config=prefill_cfg)   # prefill 用较大的 window
    rt(host_x, weight, host_out, config=decode_cfg)    # decode 用较小的 window

# 一次性路径同样支持该覆盖：
compiled(a, b, c, config=RunConfig(platform="a2a3", ring_heap=8 * 1024 * 1024))
```

在 L3 派发路径上，仅消费 `RunConfig` 的 `ring_*` 字段；编译期字段与 DFX 字段会被
忽略（它们在编译 / prepare 时设置，而非按派发设置）。

只设置你想覆盖的字段即可；其余字段保持未设置，并按上述优先级回退。

## 相关文档

- Simpler 运行时侧实现：`runtime/src/<arch>/runtime/tensormap_and_ringbuffer/`
  下的 `tensormap_and_ringbuffer` 运行时，以及
  `runtime/src/common/task_interface/call_config.h` 中的 `RuntimeEnv` /
  `CallConfig` 定义。
- Worker API 示例：`runtime/examples/workers/{l2,l3}/per_task_runtime_env/`。
- 接受单次派发 `RunConfig` 的 L3 派发入口：
  [`distributed_runner.py`](../../../python/pypto/runtime/distributed_runner.py)
  中的 `DistributedWorker.run` / `__call__` 以及 `execute_distributed`。
- 同一 `RunConfig` 上正交的运行时诊断特性：
  [03-runtime-dfx.md](03-runtime-dfx.md)。
