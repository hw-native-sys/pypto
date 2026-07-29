# 参考

PyPTO 生成代码背后的硬件与指令集资料。

当你在阅读生成的 PTO 代码、调优跨核流水，或推敲某个 pass 为何这样 lower 时阅读本章。
你所编写的语言见[用户手册](../user/index.md)；编译器如何变换它见[开发者文档](../dev/index.md)。

## PTO ISA

| 页面 | 内容 |
| ---- | ---- |
| [集群架构](pto-isa/00-cluster_architecture.md) | 1 个 Cube + 2 个伙伴 Vector 核构成的集群及其基于 flag 的同步机制 |
| [TPUSH/TPOP 指令](pto-isa/01-tpush_tpop.md) | 在同一集群内共同调度的 Cube 与 Vector InCore kernel 之间搬运 tile |
| [缓冲区管理](pto-isa/02-buffer_management.md) | TPUSH/TPOP 环形缓冲区的位置随平台而异 —— A2/A3 在 GM，A5 在消费者的片上内存 |

## 另请参阅

- [PTO 项目生态](../dev/00-ecosystem.md) —— PyPTO、PTOAS、pto-isa 与运行时如何组合。
- [PTO Codegen](../dev/codegen/00-pto_codegen.md) —— PyPTO IR 如何变成 PTO-ISA 方言的 MLIR。
- [PTOAS 算子状态矩阵](../dev/ptoas-op-status.md) —— 编译器当前会发射哪些 PTOAS 算子。
