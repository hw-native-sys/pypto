# PTO ISA

PyPTO 生成代码所面向的硬件模型与指令语义。

| 页面 | 内容 |
| ---- | ---- |
| [集群架构](00-cluster_architecture.md) | 1 个 Cube + 2 个伙伴 Vector 核构成的集群及其基于 flag 的同步机制 |
| [TPUSH/TPOP 指令](01-tpush_tpop.md) | 在同一集群内共同调度的 Cube 与 Vector InCore kernel 之间搬运 tile |
| [缓冲区管理](02-buffer_management.md) | TPUSH/TPOP 环形缓冲区的位置随平台而异 —— A2/A3 在 GM，A5 在消费者的片上内存 |

## 另请参阅

- [SkewCrossCorePipeline Pass](../../dev/passes/27-skew_cross_core_pipeline.md) —— 把跨核循环软流水到该架构上的 pass。
- [InjectGMPipeBuffer Pass](../../dev/passes/23-inject_gm_pipe_buffer.md) —— Ascend910B 上经 GM 路由的跨核 pipe workspace。
