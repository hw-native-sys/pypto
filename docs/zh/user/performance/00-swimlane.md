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
    enable_chip_swimlane=4,      # 逐任务计时，完整采集
    enable_dep_gen=True,       # 计时所要 join 的任务 DAG
    save_kernels=True,         # 保留输出目录
)
```

两者都落在本次运行的输出目录下：

```text
<work_dir>/dfx_outputs/
  chip_swimlane_records.json   逐任务时序及调度器/编排器阶段（等级 4）
  deps.json                    任务依赖边
  merged_swimlane_*.json       仅上板 —— join 之后的 trace
```

第二个开关的意义在于：后继边被**刻意不**记录进泳道图记录本身，以保持设备侧热路径干净。join 是事后在 host 上做的。

### 采集是分级的

运行时按四个等级采集，每一级在下一级之上追加：

| 等级 | 追加 |
| ---- | ---- |
| 1 `AICORE_TIMING` | AICore 逐任务 start / end |
| 2 `AICPU_TIMING` | + AICPU 打点的 dispatch / finish |
| 3 `SCHED_PHASES` | + 调度器主循环阶段记录 |
| 4 `ORCH_PHASES` | + 编排器阶段记录 |

每一级在采集器里都是实打实的门禁，不是啰嗦程度设置：等级 1 下 dispatch 与 finish 时间戳**根本不会被打**，任何后处理都恢复不出来。

`RunConfig.enable_chip_swimlane` **就是**这个等级，直接传 `0`-`4` 即可：

```python
cfg = RunConfig(platform="a2a3", enable_chip_swimlane=3,  # 调度器阶段及以下
                enable_dep_gen=True, save_kernels=True)
```

为保持调用方兼容，仍接受 `True`，它表示等级 `4`（完整采集）—— 与裸 `--enable-chip-swimlane`
请求的是同一件事；`False` 表示 `0`。等级越高采集越多、对计时的扰动也越大，所以请用能回答你
问题的最低等级。

### 不知道就会被误导的两件事

**上板时，这个开关会把你的负载跑两遍。** 转换器需要一张只有 `deps.json` 才携带的任务图，而采集本身会扰动计时 —— 所以 PyPTO 先跑一遍 dep_gen 抓图，再关掉 dep_gen 跑一遍干净的计时。**永远不要从开了泳道图的上板运行里读墙上时间**，那个数字要用另一次普通运行去取。

**在模拟器上你能拿到记录，但拿不到 merged trace。** `*sim` 平台保持单遍，只产出 `chip_swimlane_records.json` —— 模拟器还没有提供转换器需要的任务元数据。用模拟器看调度的**形状**，计时本身的问题用上板运行去看。

## 打开

### IDE 插件

[PyPTO Toolkit](https://github.com/hw-native-sys/pypto-tools) 是一个直接渲染这些文件的 VS Code 插件。在资源管理器里右键泳道图 JSON，选 **`PyPTO Toolkit：打开文件`**。

视图数量取决于采集等级：

| 视图 | 显示 | 需要等级 |
| ---- | ---- | -------- |
| Worker View | 逐核任务条 —— 主画面 | 1 |
| Scheduler View | 调度器自身的时间线 | 3 |
| AICPU Scheduler / AICPU Orchestrator | 逐迭代的调度器阶段拆解 | 3 / 4 |

第一天就值得知道的几件事：

- **点击任务**看详情面板。同级目录下有 `deps.json` 时，与它有依赖关系的任务会一并画出；*任务连线层级* 可以限制一次画出多少层依赖。
- **选中一个 SPMD 任务会同时高亮它的所有 block**，因为它们共用同一个 `func_name` 与 `task_id`。在依赖路径上只画第一个，免得视图被连线埋掉。
- **搜索**框支持按 `func_name` 或 `task_id` 模糊查找。
- **性能统计**（右上角）打开报告；点击其中的条目会跳到时间线上对应的任务。
- **观测线** —— 在二段轴上点击可放下一条带时间戳的标尺，ALT+拖动可手动测距。
- 同一个插件还能把 `passes_dump` 目录作为 **IR trace** 打开，并按「是否真的改了东西」过滤 pass。那是编译期视图而非计时视图，但它是「编译器到底对我的 kernel 做了什么」的另一半。

### Perfetto

运行时也能把记录转成可以在 [ui.perfetto.dev](https://ui.perfetto.dev) 里加载的 Chrome Trace Event JSON：

```bash
RECORDS="outputs/<run>/chip_swimlane_records.json"
DEPS_JSON="outputs/<run>/deps.json"
python -m simpler_setup.tools.swimlane_converter "$RECORDS" \
    --deps-json "$DEPS_JSON" -o out.json
```

## 怎么读

每个任务带四个时间戳，它们之间的间隙含义各不相同：

```text
dispatch ──────► start ──────► end ──────► finish
   │               │             │            │
   AICPU 写下      核开始跑      kernel       AICPU 观察到
   描述符          kernel        跑完         完成
   └── 等级 2 ──┘ └──── 等级 1 ────┘ └── 等级 2 ──┘

[dispatch, start]  = 取件延迟（每次切换约 0.8 µs）
[start, end]       = kernel 本身 —— 04 页唯一能压缩的那一段
```

等级 4 —— 也就是 `RunConfig` 给你的那一级 —— 包含全部四个时间戳以及调度器和编排器阶段。`[start, end]` 仍是任务的 AICore 执行时间；拆分 `[dispatch, start]` 至少需要等级 2，因此这里同样可以看到。

**读间隙，不是读条。** 条窄、隙宽的芯片不是 kernel 问题，而是粒度或派发问题，该去 [01](01-task-granularity.md) 与 [02](02-runtime-overhead.md) 两页。

### 给间隙一个数字

当你想把间隙量化而不是靠眼估时，运行时自带一份分析，它只回答一个问题 —— *什么时候，一个空闲的核明明有就绪的活、而调度器还没把活放上去？*

```bash
# $RECORDS 与 $DEPS_JSON 沿用上面的赋值
python -m simpler_setup.tools.sched_overhead_analysis \
    --chip-swimlane-records-json "$RECORDS" --deps-json "$DEPS_JSON"
```

它给出逐引擎与全系统的开销占 makespan 的比例、取件代价分布、AICPU 调度循环预算，以及把关键路径拆成「计算」与「调度器注入」两部分的归因。同样这些数字可以用 `swimlane_converter --overhead` 叠加成时间线上的 counter 轨。

它的调度循环部分需要**等级 ≥ 3** 的采集 —— `RunConfig(enable_chip_swimlane=3)`，或默认的完整等级 `4`。

那份报告里有两个定义值得记住，因为它们把真问题和假问题分开了：

- **没有就绪活**的空闲核**不算**开销。它的空闲是依赖图规定的 —— 那表现为并行度低，属于 [管理任务依赖](03-dependencies.md)。
- **有就绪但未派发的活**时的空闲核算开销。那是调度器没跟上，属于 [运行时开销](02-runtime-overhead.md)。

### makespan 究竟花在哪了

`sched_overhead_analysis` 回答的是一个具体问题。关键路径分析回答的是更大的那个 —— *依赖决定的下限在哪，剩下的时间被谁花掉了？*

```bash
RUN_DIR="outputs/<run>"        # 存放本次采集的那棵树
python -m simpler_setup.tools.critical_path "$RUN_DIR"
```

它会找出每一个同时含有 `chip_swimlane_records.json`、`deps.json` 与 `name_map*.json` 的目录，并在各自的 records 文件旁边写一份逐 rank 的报告。把它指向整棵 run 树，而不是单个 rank。

它给出两条路径，而结论正在于两者之差：

| 路径 | 是什么 |
| ---- | ------ |
| **Static CPM** | 按时长加权的最长链 —— 核数无限时的延迟下限 |
| **Observed** | 从最后结束的任务往回走的实际执行路径 |

每个任务的计算时间加上它前面的停顿，正好铺满 observed makespan；停顿被归为 `data-wait`（上游生产者迟到）、`core-wait`（分到的核在忙 —— 资源串行）或 `front-gap`（第一个任务开始前的启动延迟）。

由此得到本章其余部分据以分叉的判断：

| 读数 | 判断 | 该去哪 |
| ---- | ---- | ------ |
| Static CPM 接近 makespan | 依赖受限 —— 加核没用 | [03](03-dependencies.md)，以及 [01](01-task-granularity.md) 的粒度 |
| Static CPM 远低于它，`core-wait` 占主导 | 资源串行 | [01](01-task-granularity.md) |
| Static CPM 远低于它，`front-gap` 很大 | 启动与派发开销 | [02](02-runtime-overhead.md)、[06](06-host.md) |
| 计算占比高、停顿低 | 确实是计算受限 | [04](04-incore.md) |

> **引用任何数字之前，先看 tiling 那一行。** 每个 rank 都会打印 `tiling check: compute+stall = ... vs makespan ...`，它必须是 `exact`。差值非零意味着这次回溯没有铺满 makespan，逐任务的归因就是不可靠的。另有两种情况会悄悄让报告失效：family 显示为 `unknown` 或 `cid<N>` 说明 name map 没解析出来，此时 family 层面的结论毫无意义；以及跑多轮时采集只覆盖**第一轮**，makespan 里含着预热成本。
>
> **一次采集只是一个样本。** 同一份没有改动的负载采两次，停顿占比可以差好几个点。绝不要拿两个配置各采一次来做对比。

`pypto-user` 插件里的 `critical-path-analysis` skill（`claude plugin install pypto-user@pypto-skills`）会把这套流程从头跑到尾 —— 定位产物、执行上面那份校验清单、并给出解读。工具本身在 runtime 里，不需要设备、不需要构建、也不需要仓库检出：它是对别人采好的一份数据做纯后处理。

## 下一步

有了这张图之后，按本章顺序往下走 —— 粒度，然后派发开销，然后依赖图，最后才是 kernel 本身。
