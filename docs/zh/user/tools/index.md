# DFX 工具

PyPTO 能告诉你的关于它编译或运行过的程序的一切 —— 以及每条信息回答的是哪个问题。

## 概念

DFX 是可观测面：编译器注意到了什么、决定了什么，以及硬件实际做了什么。它沿一条轴分成两半，这条轴决定你在本章的哪一侧：

| 半边 | 运行于 | 代价 | 回答 |
| ---- | ------ | ---- | ---- |
| **编译期** | 不需要设备 | 几秒 | 编译器*决定*了什么 |
| **运行期** | 设备或仿真器 | 一次运行，有时两次 | 硬件*做*了什么 |

几乎每个问题在编译期一侧都更便宜，而且编译器通常在你开口前就已经回答了。从那里开始。

## 先看便宜的那两样 {#the-cheap-checks-first}

有两样东西读起来不花任何代价，而且每次编译都已经产出了：

- **`report/perf_hints.log`** —— 编译器注意到但没有拒绝的东西：低于硬件粒度的搬运、它没能 tile 的 matmul、没放下的流水深度。还会往 stderr 打一行摘要。
- **报错本身**（如果有的话）。PyPTO 区分用户错误与内部错误，这个区分直接告诉你该改自己的代码还是该提 bug —— 见[调试](00-debugging.md)。

## 目录

| 页面 | 覆盖 | 侧 |
| ---- | ---- | -- |
| [调试](00-debugging.md) | 错误类型、日志级别、pass dump、读降级后的 IR | 编译期 |
| [Torch codegen](01-torch-codegen.md) | 在 host 上跑 IR 的语义，把 IR 的错与设备的错分开 | 编译期 |
| [内存图](02-memory-map.md) | 片上放了什么、放在哪、活多久 | 编译期 |
| [IR trace](03-ir-trace.md) | 哪个 pass 改了什么 —— 整条流水线的可导航 diff | 编译期 |
| [运行期 DFX](04-runtime-dfx.md) | 五个采集开关 —— 泳道、参数 dump、PMU、依赖图、scope 统计 | 运行期 |
| [回放一次构建](05-replay.md) | 重跑、手改、重新测量一个已有的产物目录 | 运行期 |
| [In-core trace](06-incore-trace.md) | 单个 kernel 的 cycle 级、分 pipe 指令 trace | 运行期 |

## 该用哪个

四个问题覆盖大部分场景，而且它们按代价排序：

| 问题 | 工具 | 页面 |
| ---- | ---- | ---- |
| 出了什么错，错在哪？ | 错误类型、日志级别、IR dump | [调试](00-debugging.md) |
| 哪个 pass 改了我的 IR？ | 对 `passes_dump/` 跑 `pypto-ir-trace` | [IR trace](03-ir-trace.md) |
| 是 **IR** 错了，还是设备错了？ | `pypto.debug.torch_codegen` | [Torch codegen](01-torch-codegen.md) |
| 片上放了什么，活多久？ | `pypto.tools.memory_map` | [内存图](02-memory-map.md) |
| 时间去哪了？ | chip 泳道图 | [运行期 DFX](04-runtime-dfx.md) |
| kernel 把时间花在哪条 pipe 上？ | in-core 指令 trace | [In-core trace](06-incore-trace.md) |

## 产物落在哪

编译写在输出目录下，运行时写在其中的 `dfx_outputs/` 里。除 `report/` 外，这里没有一样是不问自来的。

```text
build_output/<program>_<timestamp>_<random>/
├── passes_dump/            # dump_passes=          -> memory map, IR trace
├── ptoas/                  # the .pto per InCore function, and ptoas's .cpp
├── ptoas_passes/           # dump_ptoas_passes=
├── kernels/                # the compiled device kernels
├── report/                 # always written
│   ├── perf_hints.log
│   └── pipeline_profile.*  # profiling=
├── debug/run.py            # the auto-emitted re-runner
└── dfx_outputs/            # written at RUN time, by the flags in 04-runtime-dfx
```

`compile()` 自己什么都不打印，所以要主动问它产物去了哪：

```python
compiled = kernel.compile(*args, config=RunConfig(dump_passes=PassDumpLevel.EXPLICIT))
print(compiled.output_dir)
```

> **`lower()` 什么都不落盘。** 它跑完 pass 就把 `Program` 交还给你 —— 这正是
> [torch codegen](01-torch-codegen.md) 想要的；但读 `passes_dump/` 的那三个工具需要 `compile()`。

## 参见

- [精度](../precision/index.md) —— 数值不对时这些工具服务的那条流程。
- [性能](../performance/index.md) —— 数值对但慢时的同类流程。
- [执行](../execution/index.md) —— 被观测的编译与派发面。
- [Runtime DFX Flags](../../dev/03-runtime-dfx.md) —— 运行期这一半背后的 `CallConfig` 契约。
