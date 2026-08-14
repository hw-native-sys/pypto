# 读泳道图

整块芯片的逐任务计时：每个任务什么时候被派发、它的核什么时候真正开始跑、什么时候结束。

> **前置**：一个能跑起来的 kernel。本页全是度量，不改任何东西。

## 为什么它排在最前

本章后面每一页要问的问题，泳道图都直接回答：

| 页面在问 | 泳道图显示 |
| -------- | ---------- |
| 任务是不是太小了？ | 条的宽度 vs 条之间的空隙 |
| 瓶颈是不是派发？ | `dispatch → start` 的间隙，以及有活可干时空闲的核 |
| 图是不是被串起来了？ | 本该堆叠的地方成了阶梯 |
| 是不是 kernel 本身慢？ | 一根很宽、周围没有空隙的条 —— 去 [InCore](04-incore.md) |

不看它就调优等于猜。编译期的 `report/perf_hints.log` 告诉你编译器**怀疑**了什么；泳道图告诉你**实际**发生了什么。

## 采集

两个开关，而且两个都要 —— 计时，以及它所归属的那张图：

```python
from pypto.runtime import ChipWorker, RunConfig

cfg = RunConfig(
    platform="a2a3",
    enable_l2_swimlane=True,   # 逐任务计时
    enable_dep_gen=True,       # 计时所要 join 的任务 DAG
    save_kernels=True,         # 保留输出目录
)
```

两者都落在本次运行的输出目录下：

```text
<work_dir>/dfx_outputs/
  l2_swimlane_records.json     逐任务 start / end / dispatch / finish
  deps.json                    任务依赖边
  merged_swimlane_*.json       仅上板 —— join 之后的 trace
```

第二个开关的意义在于：后继边被**刻意不**记录进泳道图记录本身，以保持设备侧热路径干净。join 是事后在 host 上做的。

### 不知道就会被误导的两件事

**上板时，这个开关会把你的负载跑两遍。** 转换器需要一张只有 `deps.json` 才携带的任务图，而采集本身会扰动计时 —— 所以 PyPTO 先跑一遍 dep_gen 取图，再关掉 dep_gen 跑一遍干净的取计时。**永远不要从开了泳道图的上板运行里读墙上时间**，那个数字要用另一次普通运行去取。

**在模拟器上你能拿到记录，但拿不到 merged trace。** `*sim` 平台保持单遍，只产出 `l2_swimlane_records.json` —— 模拟器还没有提供转换器需要的任务元数据。用模拟器看调度的**形状**，计时本身的问题用上板运行去看。

## 打开

### IDE 插件

[PyPTO Toolkit](https://github.com/hw-native-sys/pypto-tools) 是一个直接渲染这些文件的 VS Code 插件。在资源管理器里右键泳道图 JSON，选 **`PyPTO Toolkit：打开文件`**。

视图数量取决于采集等级：

| 视图 | 显示 |
| ---- | ---- |
| Worker View | 逐核任务条 —— 主画面 |
| Scheduler View | 调度器自身的时间线 |
| AICPU Scheduler / AICPU Orchestrator | 逐迭代的调度器阶段拆解 |

第一天就值得知道的几件事：

- **点击任务**看详情面板。同级目录下有 `deps.json` 时，与它有依赖关系的任务会一并画出；*任务连线层级* 可以限制一次画出多少层依赖。
- **选中一个 SPMD 任务会同时高亮它的所有 block**，因为它们共用同一个 `func_name` 与 `task_id`。在依赖路径上只画第一个，免得视图被连线埋掉。
- **搜索**框支持按 `func_name` 或 `task_id` 模糊查找。
- **性能统计**（右上角）打开报告；点击其中的条目会跳到时间线上对应的任务。
- **观测线** —— 在二段轴上点击可放下一条带时间戳的标尺，ALT+拖动可手动测距。
- 同一个插件还能把 `passes_dump` 目录作为 **IR trace** 打开，并按「是否真的改了东西」过滤 pass。那是编译期视图而非计时视图，但它是「编译器到底对我的 kernel 做了什么」的另一半。

> **要预期到的一个命名差异。** 插件文档里这个文件叫 `chip_swimlane_records*.json`；本仓库所 pin 的运行时写出的是 `l2_swimlane_records.json`。如果右键菜单没有提供打开你的文件，先查这个差异。

### Perfetto

运行时也能把记录转成可以在 [ui.perfetto.dev](https://ui.perfetto.dev) 里加载的 Chrome Trace Event JSON：

```bash
python -m simpler_setup.tools.swimlane_converter <records>.json \
    --deps-json <deps>.json -o out.json
```

## 怎么读

每个任务带四个时间戳，它们之间的间隙含义各不相同：

```text
dispatch ──────► start ──────► end ──────► finish
   │               │             │            │
   AICPU 写下      核开始跑      kernel       AICPU 观察到
   描述符          kernel        跑完         完成

[dispatch, start]  = 取件延迟（每次切换约 0.8 µs）
[start, end]       = kernel 本身 —— 04 页唯一能压缩的那一段
```

**读间隙，不是读条。** 条窄、隙宽的芯片不是 kernel 问题，而是粒度或派发问题，该去 [01](01-task-granularity.md) 与 [02](02-runtime-overhead.md) 两页。

### 给间隙一个数字

当你想把间隙量化而不是靠眼估时，运行时自带一份分析，它只回答一个问题 —— *什么时候，一个空闲的核明明有就绪的活、而调度器还没把活放上去？*

```bash
python -m simpler_setup.tools.sched_overhead_analysis \
    --l2-swimlane-records-json <records>.json --deps-json <deps>.json
```

它给出逐引擎与全系统的开销占 makespan 的比例、取件代价分布、AICPU 调度循环预算，以及把关键路径拆成「计算」与「调度器注入」两部分的归因。同样这些数字可以用 `swimlane_converter --overhead` 叠加成时间线上的 counter 轨。

那份报告里有两个定义值得记住，因为它们把真问题和假问题分开了：

- **没有就绪活**的空闲核**不算**开销。它的空闲是依赖图规定的 —— 那表现为并行度低，属于 [管理任务依赖](03-dependencies.md)。
- **有就绪但未派发的活**时的空闲核算开销。那是调度器没跟上，属于 [运行时开销](02-runtime-overhead.md)。

## 下一步

有了这张图之后，按本章顺序往下走 —— 粒度，然后派发开销，然后依赖图，最后才是 kernel 本身。
