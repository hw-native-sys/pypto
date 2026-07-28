# 性能调优：循环与流水线

> **状态：** 骨架 —— 仅提纲；正文/示例待定。

## `pl.pipeline`

> 逐循环 opt-in。非 mixed 与 mixed（跨核）循环的效果不同 —— **两者都要讲清楚**。

### 非 mixed（同核）循环

- 在单核内重叠 load/compute/store 阶段（GM→L1、L1→L0、嵌套 matmul stage 循环 ——
  无 `tpush`/`tpop`）。
- `LowerPipelineLoops` 复制各阶段（`stage=F`）。

### mixed（跨核 C→V→C）循环

- `SkewCrossCorePipeline` 把 cube(AIC) 与 vector(AIV) 两半错开（skew），使两个核
  重叠而非互相等待。
- pass 默认开启，但只作用于你标了 `pl.pipeline(stage=F)` 且 `F > 1` 的循环。
- `stage` → skew 深度 `D = max(2, stage - 1)`（默认深度 2）。
- 依角色而定：生产者角色单次往返 → 真正的 producer skew；消费者角色 / 多次往返 /
  动态边界 / trip < 2 → 降级为 Sequential（此时重叠靠对端核的 skew）。

### mixed 流水不理想时：手动跨核同步

- `pl.system.sync_set(event_id, pipe=..., core_type="aic"|"aiv")`
- `pl.system.sync_wait(event_id, pipe=..., core_type="aic"|"aiv")`
- 当自动 skew 无法产生期望的重叠时，手动控制 Cube/Vector 跨核事件。

## `pl.unroll`

- 展开因子；权衡（代码体积 vs 调度自由度）。

## `pl.parallel`

- 标记独立迭代以并行执行。

## 验证效果

- 泳道图 → [DFX › 泳道图](../dfx/01-swimlane.md)
