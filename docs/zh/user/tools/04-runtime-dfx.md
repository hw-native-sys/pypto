# 运行期 DFX

五个互相独立的采集开关。每个记录什么、落在哪、用什么打开。

## 概念

本页所有东西都要花掉一次运行。编译器已经告诉过你它*决定*了什么（见[先看便宜的那两样](index.md#the-cheap-checks-first)）；这些开关记录的是硬件*做*了什么。

它们是 `RunConfig` 上五个完全独立的开关，可以任意组合。**开启其中任意一个都会强制 `save_kernels=True`**，这样输出目录在运行结束后仍然存在，产物还在。

所有产物落在同一个目录下：

```text
<work_dir>/dfx_outputs/
├── chip_swimlane_records.json     enable_chip_swimlane
├── merged_swimlane_*.json         (onboard only — the joined trace)
├── deps.json                      enable_dep_gen
├── args_dump/                     enable_dump_args
├── pmu.csv                        enable_pmu
└── scope_stats/scope_stats.jsonl  enable_scope_stats
```

## 快速上手

最先要的是两个开关，而且要一起开 —— 时序，以及时序所属的那张任务图：

```python
from pypto.runtime import RunConfig

cfg = RunConfig(
    platform="a2a3",
    enable_chip_swimlane=4,     # per-task timing, full collection
    enable_dep_gen=True,        # the task DAG the timing is joined against
    save_kernels=True,          # forced anyway; be explicit about wanting the directory
)
out = kernel(a, b, torch.zeros(...), config=cfg)
```

从 pytest 用同样两个：

```bash
pytest tests/st/runtime/ --platform a2a3 --enable-chip-swimlane --enable-dep-gen
```

## 机制

### 开关矩阵

| `RunConfig` 字段 | pytest flag | `dfx_outputs/` 下的产物 | 用什么打开 |
| ---------------- | ----------- | ----------------------- | ---------- |
| `enable_chip_swimlane: int` | `--enable-chip-swimlane`（= `4`）/ `--chip-swimlane-level N` | `chip_swimlane_records.json` | PyPTO Toolkit；板上会自动产出 `merged_swimlane_*.json` |
| `enable_dep_gen: bool` | `--enable-dep-gen` | `deps.json` | `python -m simpler_setup.tools.deps_viewer` |
| `enable_dump_args: int` | `--dump-args [LEVEL]`（裸写 = `1`） | `args_dump/` | `python -m simpler_setup.tools.dump_viewer` |
| `enable_pmu: int` | `--enable-pmu [N]`（裸写 = `2`） | `pmu.csv` | 任意 CSV 阅读器 |
| `enable_scope_stats: bool` | `--enable-scope-stats` | `scope_stats/scope_stats.jsonl` | `python runtime/simpler_setup/tools/scope_stats_plot.py` |

**没有一个渲染器是自动跑的。** Graphviz 对上千节点的图做布局可能要几分钟，而把它放在派发热路径上跑，曾导致外层调度器把整个作业树 SIGKILL 掉。运行器只会把渲染命令打出来，你想看图时自己跑。

### 泳道图：采集是分级的

`enable_chip_swimlane` 是**等级**，不是开关：

| 等级 | 增加 | 解锁 |
| ---- | ---- | ---- |
| `0` / `False` | — | 关 |
| `1` | AICore 每任务 start / end，以及任务记录缓冲 | 每任务泳道 |
| `2` | + AICPU 打点的 dispatch / finish | `[dispatch, start]` 取件间隙 |
| `3` | + 调度器主循环阶段记录 | `sched_overhead_analysis`、Toolkit 的 Scheduler View |
| `4` / `True` | + 编排器阶段记录 | Toolkit 的 AICPU Orchestrator 视图 |

**每一级都是采集器里的真实 guard，不是冗余度设置。** 等级 1 根本不打 dispatch / finish 时间戳，任何后处理都恢复不出来 —— 「先便宜地采，回头再深入分析」是行不通的。反过来，等级越高对时序的扰动越大。取能回答你问题的**最低**等级。

`True` 表示等级 `4`，与裸写的 `--enable-chip-swimlane` 一致。超范围的等级会让 `RunConfig` 抛 `ValueError`。

打开之后*该看什么*，见[读泳道图](../performance/00-swimlane.md)。

### 泳道图：板上会把你的 workload 跑两遍

converter 需要把每任务时序与一张**只有 `deps.json` 才带**的任务图做 join —— 设备热路径已不再记录每任务的 fanout。但 dep_gen 采集会扰动泳道所要测量的那个时序。于是两次采集来自两次独立运行，而 PyPTO 替你做了这个拆分：

1. **抓图那遍** —— 只开 dep_gen，产出 `deps.json`。L2 上这一遍在独立子进程里跑，因为运行时不能可靠回收采集器申请的 SVM host-register 映射，同进程内第二次 DFX 运行会撞上注册上限。这一遍是 best-effort：失败则记一条警告，计时那遍照常跑，泳道退化为匿名的 `task(rXtY)`。
2. **计时那遍** —— 泳道（以及其他对时序敏感的 DFX），dep_gen 强制关闭，产出被报告的那份 `chip_swimlane_records.json`。

值得记住的后果：

- **绝不要从开了泳道的板上运行里读墙钟时间。** 单独跑一次干净的，或用 [`benchmark`](../performance/index.md#before-you-tune-anything)。
- **两遍都会真的执行程序**，且两遍之间不会恢复可变参数。
- 额外加 `--enable-dep-gen` 不会改变这两遍的行为 —— 抓图那遍已经产出了 `deps.json`。它只是让运行额外打印 `deps_viewer` 的提示。
- **仿真器平台（`*sim`）保持单遍**，只产出 `chip_swimlane_records.json`；仿真器还没有 converter 需要的任务元数据。用仿真器看调度的*形状*，时序本身的问题上板看。

### 参数 dump：标记要看的张量，不要全量拿

`enable_dump_args` 是等级：`0` 关，`1` 部分，`2` 全量。

**优先用部分。** 等级 `2` 会写下每个任务的每一个 binding；在大 workload 上这会打满 host 侧的 dump 收集器（约 42 MB/s 排空速率），AICPU 随后被 STARS op-execute 超时杀掉 —— 一个 1 GB 的 KV-cache binding 填队列的速度远快于排空。

部分 dump 只采集你标记的东西，通过两个与 `deps=` 完全对称的面：

```python
# Declarative marker — every SUBSEQUENT dispatch consuming that exact value dumps it.
pl.dump_tag(q)
pl.dump_tag(out)
out = self.qk_pv(q, k_cache, out)        # q and out dumped; k_cache filtered out

# Explicit kwarg — this one launch only.
with pl.manual_scope():
    out, tid = pl.submit(self.qk_pv, q, k_cache, out, deps=[prev], dumps=[q, out])
```

同一个 `dumps=[...]` kwarg 也可以写在派发 scope 上 —— `pl.at(...)`、`pl.spmd(...)`、
`pl.cluster(...)` 或 `pl.graph(...)` —— 标记该 scope 启动的 task。

标记按 **Var 身份追踪，绝不按名字**，所以它能穿过 SSA、内联和 codegen —— 也因此，被重新绑定或变换过的值**不**被覆盖：

| 你写的 | 覆盖吗？ |
| ------ | -------- |
| `pl.dump_tag(q)` 然后 `self.k(q, ...)` | 是 |
| `pl.dump_tag(q)` 然后 `q = self.foo(q)` 然后 `self.k(q, ...)` | **否** —— 重新绑定的结果是新值，要重新打标 |
| `pl.dump_tag(q)` 然后 `q2 = pl.reshape(q)` 然后 `self.k(q2, ...)` | **否** —— 给 kernel 实际收到的那个值打标 |
| 在普通的 `self.kernel(...)` 调用上写 `dumps=` | **否** —— 抛 `ParserTypeError`；改用 `pl.dump_tag` 或 `pl.submit(..., dumps=[...])` |
| 在 InCore / AIC / AIV 函数体里写 `pl.dump_tag` | **否** —— 解析期抛 `ParserSyntaxError`；放到外层的 Orchestration 或 Inline 函数里 |

dump 关闭时标记是惰性的，全量 dump 下则无关紧要。完整的限制表见 [Runtime DFX Flags](../../dev/03-runtime-dfx.md#limitations)。

### PMU

```python
RunConfig(enable_pmu=2)     # 0 = off; 2 = PIPE_UTILIZATION; 4 = MEMORY
```

一次运行一个事件类型，写入 `pmu.csv`。以比 [in-core trace](06-incore-trace.md) 更粗的粒度回答「各 pipe 在干什么」，而且不需要仿真器。

### Scope 统计

per-scope 的 heap / task_window / tensormap 环形缓冲填充**峰值**，每个 scope 一条 JSONL 记录：

```bash
python runtime/simpler_setup/tools/scope_stats_plot.py \
    <work_dir>/dfx_outputs/scope_stats/scope_stats.jsonl
```

这是 ring 定容背后的测量 —— 在动 `ring_task_window`、`ring_heap`、`ring_dep_pool` 之前先读它。见[内存](../performance/05-memory.md)。

### 查看器里的 kernel 名字

默认情况下泳道和依赖查看器用数字 id 标注任务。只要开了泳道或 dep_gen，PyPTO 就会从 `kernel_config.py` 合成 `dfx_outputs/name_map_<case>.json`，两个工具都会自动拾取 —— 所以任务直接显示真实 kernel 名，不需要手工步骤。

### 分布式运行

L3 按每次派发给前缀加命名空间，因为一块芯片在一次 host 编排里可能收到多次派发，否则它们会互相覆盖：

```text
<work_dir>/dfx_outputs/rank0/d0/     # rank 0, its 0th dispatch
<work_dir>/dfx_outputs/rank0/d1/
<work_dir>/dfx_outputs/rank1/d0/
```

每个叶子目录里放着上面那张表里的扁平产物，所以本页内容在单次派发目录内原样适用。

## 边界情况

| 现象 | 原因 | 处理 |
| ---- | ---- | ---- |
| **根本没有 `dfx_outputs/`** | 没开任何开关，或运行在派发前就失败了 | 开任一开关都会强制 `save_kernels`；先确认运行真的到了设备 |
| **墙钟时间比预期差很多** | 板上泳道把 workload 跑了两遍 | 用单独一次干净运行来测 |
| **没有 `merged_swimlane_*.json`** | 仿真器平台 | 符合预期 —— `*sim` 只产出 records |
| **泳道全是匿名的 `task(rXtY)`** | 抓图那遍失败了，没有 `deps.json` | 从日志里那条警告去定位 —— 加 `--enable-dep-gen` 不会重跑任何东西，那一遍本来就会自己跑 |
| **AICPU 中途被杀，STARS 超时** | 大 workload 上开了 `enable_dump_args=2` | 降到等级 `1`，并用 `pl.dump_tag` 标记需要的张量 |
| **打了标的张量没出现在 dump 里** | 标记按 Var 身份追踪，而该值被重新绑定或变换过 | 给 kernel 实际收到的那个值打标 |
| **`deps_viewer --format html` 卡住** | 超大图上跑 Graphviz `dot` | 换 `--engine sfdp`（O(N log N)，可扩展到 1 万+ 节点） |
| **`enable_l2_swimlane` 的 `DeprecationWarning`** | 旧拼写 | 改名为 `enable_chip_swimlane`；取值与语义不变 |

## 参见

- [读泳道图](../performance/00-swimlane.md) —— 打开之后该看什么。
- [回放一次构建](05-replay.md) —— 本页每个开关在回放路径上原样适用。
- [In-core trace](06-incore-trace.md) —— 再往下一层，深入单个 kernel 内部。
- [内存](../performance/05-memory.md) —— `scope_stats` 测量的那些 ring。
- [精度](../precision/00-workflow.md) —— `enable_dump_args` 在收敛流程里的位置。
- [Runtime DFX Flags](../../dev/03-runtime-dfx.md) —— `CallConfig` 契约、完整的 dump-tag 限制表，以及废弃别名。
