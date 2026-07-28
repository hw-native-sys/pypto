# 性能调优：工作流

**症状：** 你的 kernel 正确但太慢。本页是调优循环 —— 测量、找瓶颈、应用匹配的旋钮、
再测量。

## 黄金法则：先测量

绝不盲调。动任何旋钮前，先 profile 找出时间实际去向
（[测量收益](05-measuring-impact.md)）：

- **Benchmark** —— `pypto.runtime.benchmark(compiled, args)` 给出端到端设备墙钟这个
  总数字（多次 launch 的稳态）。先看它判断一次改动*是否*有效；用下面的视角看*为什么*。
- **编译期 profiling** —— `ir.compile(..., profiling=True)`（或 `PYPTO_COMPILE_PROFILING=1`
  环境变量）把 `output_dir/report/pipeline_profile.{txt,json}` 写出。注意它测的是
  **编译**墙钟（逐 pass、逐 kernel codegen），即“为什么*编译*慢” —— 片上 kernel
  性能请用下面两个工具。
- **In-core profiling** —— 通过昇腾 msprof op-simulator 得到 cycle 精确的逐 kernel
  trace（`incore-profiling` skill）。这是你的**核内**视角 —— 一个 kernel/task *内部*
  发生了什么，包括 cube/vector 重叠。
- **设备侧泳道图** —— `enable_l2_swimlane` 捕获逐 task 时间线，显示不同 task 之间是
  停顿还是重叠（[DFX › 泳道图](../dfx/01-swimlane.md)）。这是你的**核间**视角。
- **Codegen 对比** —— 跨分支 diff `.pto` / pass dump，看一次改动实际产出了什么
  （`compare-codegen` skill）。

然后**一次只改一个旋钮**并重新测量。保留前后数字。

## 两个维度：核内 vs 核间

调优分成两个问题，各有自己的旋钮和自己的测量视角：

| 维度 | 目标 | 旋钮章节 | 用什么测量 |
| ---- | ---- | -------- | ---------- |
| **核内** | 让单个 kernel（一个 task）跑满 —— cube+vector 重叠、循环流水、内存 | 循环与流水线、Mixed kernel、内存放置 | in-core profiling（msprof） |
| **核间** | 让不同 task / 核重叠 | SPMD 多 block、依赖与分发 | 泳道图（逐 task 时间线） |

> **Mixed kernel**（1 cube + 2 vector，共同调度）是**一个 task** —— 在泳道图上是
> 单个块。它内部的 cube/vector 重叠（包括 `pl.pipeline` 的 C→V→C skew）因此属于
> **核内**，从 in-core profile 看，而非泳道图。

## 调优循环

```text
太慢 → 先测量（章节 05）
│
├── 核内（单 kernel/task 利用不足；看 in-core profile）
│   ├─ 同核 load/compute/store 未重叠？        → 章节 01
│   │     pl.pipeline · pl.unroll · pl.parallel
│   ├─ cube(AIC) 与 vector(AIV) 未重叠？        → 章节 02
│   │     mixed kernel: pl.split(SplitMode) · pl.split_aiv · sync_set/wait
│   └─ 访存受限 / 数据在错误层级？             → 章节 04
│         memory-space 提示 · L1 复用 · L0 分块 · 512B 对齐
│
└── 核间（不同 task/核 空闲或串行；看泳道图）
    ├─ 工作未铺到多个 block / 核？             → 章节 02
    │     片上: pl.spmd · pl.cluster · pl.at · syncall
    └─ 任务本可重叠却串行？                     → 章节 03
          deps= · no_dep · dummy_task · allow_early_resolve · predicate
```

## 旋钮索引

**核内**（用 in-core profiling 测量）：

| 关注点 | 旋钮 | 章节 |
| ------ | ---- | ---- |
| 循环 / 流水线重叠（同核） | `pl.pipeline`、`pl.unroll`、`pl.parallel` | [01](01-loop-pipeline.md) |
| Mixed kernel —— cube + vector 重叠（一个 task） | `pl.split`(`SplitMode`)、`pl.split_aiv`、`pl.system.sync_set`/`pl.system.sync_wait` | [02](02-split-parallel.md) |
| 内存放置 | `target_memory` / `pl.Mem` 提示（L1 复用与 L0 分块自动） | [04](04-memory-placement.md) |

**核间**（用泳道图测量）：

| 关注点 | 旋钮 | 章节 |
| ------ | ---- | ---- |
| 片上 SPMD 多 block | `pl.spmd`、`pl.cluster`、`pl.at`、`pl.spmd_submit`、`pl.system.syncall` | [02](02-split-parallel.md) |
| 依赖与分发控制 | `deps=`、`pl.no_dep`、`dummy_task`、`allow_early_resolve`、`predicate`、`pl.manual_scope`/`pl.submit`、`TaskId` | [03](03-dependency-dispatch.md) |

**共用：** [测量收益](05-measuring-impact.md) —— benchmark（总墙钟）、编译期 profiling、
in-core msprof、泳道图、codegen 对比。

## 如何阅读本部分

每个旋钮章节按 **何时用 / 如何写 / 效果 / 如何验证** 组织，便于你直接跳到测量指向的
那个旋钮。两个核间家族相互作用：

- 片上分发（[章节 02](02-split-parallel.md) 的 `pl.spmd`、`pl.at`、`pl.spmd_submit`）
  接受[章节 03](03-dependency-dispatch.md)所述的*调度*参数（`deps=`、
  `allow_early_resolve=`、`predicate=`）。
- 改动依赖后，先通过 [DFX › 诊断渲染](../dfx/02-diagnostics-render.md)
  （`deps.json` → HTML）重新检查任务图，再相信墙钟数字。

## 参见

- 开发者参考：[`dev/01-compile-profiling.md`](../../../dev/01-compile-profiling.md)、[`dev/03-runtime-dfx.md`](../../../dev/03-runtime-dfx.md)
