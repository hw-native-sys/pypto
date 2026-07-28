# DFX 功能：诊断渲染

> **状态：** 草稿骨架。把原始诊断产物渲染成可浏览的 HTML。

## 依赖图（`deps.json` → HTML）

_TODO —— 如何渲染任务依赖图并解读（节点=任务、边=依赖、如何发现串行化）。_

## Scope 统计（`scope_stats.jsonl` → HTML）

_TODO —— 逐 scope 的时序/占用视图；如何解读。_

## 参见

- [标志矩阵](00-flag-matrix.md) —— 哪些标志产出这些文件
- [性能 › 依赖与分发](../perf/03-dependency-dispatch.md) —— 根据发现采取行动
- 开发者参考：[`dev/03-runtime-dfx.md`](../../../dev/03-runtime-dfx.md)
