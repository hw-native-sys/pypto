# 单任务 Ring 尺寸配置（Per-Task Ring Sizing）

PyPTO 通过 [`RunConfig`](../../../python/pypto/runtime/runner.py) 上的三个可选
覆盖项暴露 Simpler 的单任务 ring 尺寸配置。它们让你在单次 `run()` 调用中为
运行时的单任务 ring 资源设置尺寸，而无需改动已编译产物或任何全局状态。

运行时把任务启动相关资源保存在 *ring buffer*（环形缓冲）中。每个覆盖项与
Simpler 的 `CallConfig.runtime_env` 上的同名字段一一对应，并且按 **每次 L2 任务
提交** 生效 —— 即每次 `run()` 调用（在 L3 派发中则是每个 chip），因此同一 kernel
的不同提交可以使用不同的 ring 尺寸。

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

只设置你想覆盖的字段即可；其余字段保持未设置，并按上述优先级回退。

## 相关文档

- Simpler 运行时侧实现：`runtime/src/<arch>/runtime/tensormap_and_ringbuffer/`
  下的 `tensormap_and_ringbuffer` 运行时，以及
  `runtime/src/common/task_interface/call_config.h` 中的 `RuntimeEnv` /
  `CallConfig` 定义。
- Worker API 示例：`runtime/examples/workers/{l2,l3}/per_task_runtime_env/`。
- 同一 `RunConfig` 上正交的运行时诊断特性：
  [03-runtime-dfx.md](03-runtime-dfx.md)。
