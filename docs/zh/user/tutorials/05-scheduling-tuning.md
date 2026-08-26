# 调度调优

搞清运行时拿你的图做了什么，然后凭证据去改。

> **前置**：[塑形任务图](04-task-graph.md)。

## 你要做的东西

一套能套用到自己 kernel 上的循环：**看图 → 看并发 → 看资源水位 → 改一处 → 再测一次。**

第 1–4 步都是各写出一个文件的 `RunConfig` 开关，其中没有任何一步需要你去改造 kernel。第 5、6 步则是改变工作本身的组织方式，不是观测。

## 四个观测点

| 问题 | 开关 | 产物 |
| ---- | ---- | ---- |
| 图长成什么样？ | `enable_dep_gen=True` | `<work_dir>/dfx_outputs/deps.json` |
| 任务真的重叠了吗？ | `enable_chip_swimlane=True` | `<work_dir>/dfx_outputs/chip_swimlane_records.json` |
| 运行时的环是否接近满？ | `enable_scope_stats=True` | `<work_dir>/dfx_outputs/scope_stats/scope_stats.jsonl` |
| 哪条 pipe 是瓶颈？ | `enable_pmu=2` | `<work_dir>/dfx_outputs/pmu.csv` |

它们可以组合 —— 一次运行可以同时采集多项。按顺序来；每一项回答的问题都是下一项的前提。

## 第 1 步：图是你想要的那张吗

```python
from pypto.runtime import RunConfig

kernel(a, b, out, config=RunConfig(platform="a2a3sim", enable_dep_gen=True))
```

然后渲染它：

```bash
python -m simpler_setup.tools.deps_viewer <work_dir>/dfx_outputs/deps.json --format html
```

viewer 默认输出文本，要图视图得传 `--format html`。

这是第一件要查的事，因为后面每一项度量都在它的下游。两种读法值得掌握：

- **该扇出的地方是一条链** —— 一条并非真实依赖的推导边正在把你串行化。回到 [上一页第 3 步](04-task-graph.md)。
- **该是链的地方是扇出** —— 没有任何东西在给必须有序的任务定序。这是潜伏的竞态，而且它不会每次都表现为错误答案。

## 第 2 步：它们真的重叠了吗

一张并行的图并不保证并行的执行。swimlane 给出逐任务时序：

```python
kernel(a, b, out, config=RunConfig(platform="a2a3sim", enable_chip_swimlane=True))
```

> **模拟器注意事项：** 在 `*sim` 平台上这是单趟的，只产出 `chip_swimlane_records.json`。合并后的 `merged_swimlane_*.json` 视图会被**有意跳过**，因为模拟器尚未提供转换器所需的任务元数据。在真机平台上同一个开关会把负载**跑两遍** —— 先一趟 dep_gen 采集图，再一趟干净的计时 —— 因为采集会扰动时序。

第二点对基准测试很重要：不要从开了 swimlane 的真机运行里读挂钟时间。

## 第 3 步：环是不是瓶颈

运行时用环来存放在飞的任务。如果某个环饱和了，往图里加并行度不会有任何收益：

```python
kernel(a, b, out, config=RunConfig(platform="a2a3sim", enable_scope_stats=True))
```

```bash
python runtime/tools/scope_stats_plot.py <work_dir>/dfx_outputs/scope_stats/scope_stats.jsonl
```

它按作用域报告三个环的峰值 —— **task_window**（在飞任务槽）、**heap**（输出存储）、**tensormap**。峰值顶到容量就是该调大那个环的信号；峰值远低于容量则说明环不是你的问题。

## 第 4 步：改一处

每个环都有对应的覆盖项。它们是逐次调用生效的，所以你可以不重新编译就扫参：

| 旋钮 | 单位 | 约束 |
| ---- | ---- | ---- |
| `ring_task_window` | 在飞任务槽数 | 2 的幂，`>= 4` |
| `ring_heap` | **字节** | 2 的幂，`>= 1024` |
| `ring_dep_pool` | 依赖边容量 | `[4, INT32_MAX]` |
| `aicpu_thread_num` | AICPU 线程数 | 默认沿用编译期的 `RUNTIME_CONFIG` |

```python
cfg = RunConfig(platform="a2a3sim", ring_task_window=64, ring_heap=1 << 20)
```

每一项都接受一个标量（广播到全部四个 scope-depth 环）或恰好 4 个 int 的列表，分别设定环 0..3，其中 `0` 表示该环保持默认。默认值 `None` 表示不设该字段，运行时回落到它的编译期默认值；旧的进程级 `PTO2_RING_*` 环境变量已经退役，不再被读取。

`ring_heap` 以字节计而 `ring_task_window` 以槽计，是最容易犯的错：`ring_heap=64` 不是 64 个缓冲区，而是 64 字节，会因为不足 1024 而被拒。

## 第 5 步：任务粒度

上面那些旋钮调的是机器本身的尺寸。更大的杠杆通常是**一个任务承载多少工作**：

- **太细** —— 逐任务的调度开销占主导，task_window 饱和在记账而非工作上。
- **太粗** —— 任务数少于核数，无论环怎么调，图都填不满机器。

SPMD 派发上的 `sync_start=True` 要求所有 block 原子启动。它换来跨 block 明确定义的起点，代价是失去让任何 block 提前开始的能力 —— 所以 `sync_start` 的任务自身不能被逐 block 预置，不过给它标上 `allow_early_resolve=True` 仍能让它的**消费者**预置。

## 第 6 步：别把 setup 付两遍

worker 的 setup 是按 worker 而非按 program 计的。若干 program 注册到同一个 worker 上就能共享它，这样第一次之后的每次运行都省掉一整次 setup —— 形态见 `examples/runtime/multi_program_kv_cache.py`（一个 prefill 与一个 decode program 共享一份 KV cache 和一个 worker）。

## 调错误答案，而不是调慢

同一族里还有两个开关，面向正确性而非速度：

- **`enable_dump_args=1`** 只 dump 你用 `pl.dump_tag(t)`（或 `pl.submit(..., dumps=[...])`）标记的张量，落到 `<work_dir>/dfx_outputs/args_dump/`。用 `python -m simpler_setup.tools.dump_viewer` 查看。
- **`enable_dump_args=2`** dump 每个任务的全部输入输出。

> **致命陷阱：** 在大负载上做全量 dump 会把主机侧收集器打满（约 42 MB/s 的排空速率），并让 AICPU 被 STARS op-execute 超时杀掉。优先用等级 1 加上对具体张量的 `pl.dump_tag(t)`。

## 这个循环

```text
deps.json      →  图对不对？        →  改边      （04-task-graph）
swimlane       →  重叠了吗？        →  改粒度
scope_stats    →  某个环饱和了吗？  →  调大那个环
pmu.csv        →  哪条 pipe 是上限？→  改 kernel（02、03）
```

每改一处就重测一次，且一次只改一处 —— 这四项观测彼此不独立，同时改两处通常会让你无法归因。

## 参见

- [塑形任务图](04-task-graph.md) —— 边从哪来。
- [混合 kernel](03-mixed-kernel.md) —— 当某个单元是瓶颈时的解法。
- [任务与定序](../tasks/index.md) —— 定序接口的参考资料。
