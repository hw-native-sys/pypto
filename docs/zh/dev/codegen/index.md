# 代码生成

把 PyPTO IR lower 成已编译程序的两类产物：设备侧 kernel，以及启动它们的主机侧编排代码。

两个生成器遵循同一条设计原则 —— 从 IR 到生成代码的**严格 1:1 映射**。任何需要做决策的
地方，都已由某个 pass 决定完毕。

| 页面 | 内容 |
| ---- | ---- |
| [PTO Codegen](00-pto_codegen.md) | 从 PyPTO IR 生成 PTO-ISA 方言的 MLIR |
| [编排代码生成](01-orchestration_codegen.md) | 生成向运行时提交任务的主机侧 C++ |

## 另请参阅

- [Passes](../passes/index.md) —— codegen 运行前必须成立的全部前提。
- [PTO ISA 参考](../../reference/index.md) —— 生成代码所使用的指令语义。
- [PTOAS 算子状态矩阵](../ptoas-op-status.md) —— 可供发射的 PTOAS 算子范围。
