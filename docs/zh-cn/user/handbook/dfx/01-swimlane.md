# DFX 功能：泳道图

> **状态：** 骨架 —— 仅提纲；正文/示例待定。
> 在硬件上捕获的逐 task 时间线。本文重点：**怎么看**。

## 开启与输出（简）

- 开启：`enable_l2_swimlane`（`RunConfig`）/ `--enable-l2-swimlane`（pytest）。
- 板上执行两次（未计时图 + 计时）；`*sim` 单次。
- `l2_swimlane_records.json` → `swimlane_converter` → `merged_swimlane_*.json`
  （`--func-names` 显示可读任务名）。

## 怎么看泳道图（重点）

- 泳道是什么（核）、块是什么（task）。Mixed kernel 是**一个** task —— 其内部的
  cube/vector 重叠*不*在这里（用 in-core profiling 看）。
- 看什么：空隙（停顿）、串行链、跨核不均衡。
- 去哪里调：任务串行 → [依赖与分发](../perf/03-dependency-dispatch.md)；
  跨核不均衡 → [切分与并行](../perf/02-split-parallel.md) 的 SPMD。
- 待补：查看器 / 如何打开 `merged_swimlane_*.json`。

## task 字段解读（重点）

- 待补：逐一列出 task 块上显示的每个字段及其含义
  （name / id、起止 / 时长、engine / pipe、依赖……）。

## L0–L4 泳道图能力（重点）

- 待补：各级（L0 / L1 / L2 / L3 / L4）分别捕获什么、粒度如何 —— 每级能看到与
  看不到什么。

## 参见

- [标志矩阵](00-flag-matrix.md)
- [性能 › 测量收益](../perf/05-measuring-impact.md)
- 开发者参考：[`dev/03-runtime-dfx.md`](../../../dev/03-runtime-dfx.md)
