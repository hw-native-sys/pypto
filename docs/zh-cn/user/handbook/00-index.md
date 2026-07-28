# 功能手册

任务导向的指南，将 PyPTO 现有工具串成工作流。每章遵循
**症状 → 工具 → 步骤 → 如何读输出**，并交叉链接参考文档而非重复。若你已知症状，
用下方索引；否则先读各支柱的工作流页。

## 症状 → 工具 索引

| 症状 | 前往 |
| ---- | ---- |
| 输出数值错误 / 与 torch 发散 | [精度 › 工作流](precision/00-workflow.md) |
| 需要定位第一个 IR 发散的 pass | [精度 › Pass-IR 二分](precision/02-pass-ir-bisection.md) |
| 想对比某个张量的设备值与黄金值 | [精度 › 选择性 dump](precision/03-selective-dump.md) |
| kernel 正确但太慢 | [性能 › 工作流](perf/00-workflow.md) |
| 循环 / 流水线未重叠 | [性能 › 循环与流水线](perf/01-loop-pipeline.md) |
| 核利用率低 / 想并行 | [性能 › 切分与并行](perf/02-split-parallel.md) |
| 任务本可重叠却串行 | [性能 › 依赖与分发](perf/03-dependency-dispatch.md) |
| 数据在错误的内存层级 | [性能 › 内存放置](perf/04-memory-placement.md) |
| 需要测量时间去向 | [性能 › 测量收益](perf/05-measuring-impact.md) |
| 想看逐 task 时间线 / 哪个核在停顿 | [DFX › 泳道图](dfx/01-swimlane.md) |
| 想检查调度 / 依赖图 | [DFX › 诊断渲染](dfx/02-diagnostics-render.md) |
| 需要详细的编译 / 运行日志 | [DFX › 日志](dfx/04-logging.md) |
| 从已有 `build_output` 复现 | [DFX › 回放](dfx/03-replay.md) |

## 三大支柱

1. **[精度定位](precision/00-workflow.md)** —— “我的输出错了 → 如何二分定位发散点”。
2. **[性能调优语法](perf/00-workflow.md)** —— DSL 旋钮，*何时*、*如何*应用，以及如何测量效果。
3. **[DFX 功能](dfx/00-flag-matrix.md)** —— 诊断、渲染、回放与日志。
