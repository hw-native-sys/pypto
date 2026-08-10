# 开发者文档

PyPTO 的构成：IR、pass 流水线、代码生成，以及围绕它们的基础设施。

本章面向**开发编译器本身**的人。如果你是在编写 PyPTO 程序，请从[用户手册](../user/index.md)开始。

## 子章节

| 章节 | 内容 |
| ---- | ---- |
| [IR](ir/index.md) | 节点层次、类型系统、算子、builder、parser、序列化、结构化比较 |
| [Passes](passes/index.md) | pass 框架与默认流水线中的每个 pass，按执行顺序编号 |
| [语言](language/index.md) | Python DSL 语法规范与外部 C++ kernel 接入 |
| [代码生成](codegen/index.md) | 把 IR lower 成 PTO-ISA 方言 MLIR 与编排层 C++ |
| [后端](backend/index.md) | 通过 `BackendHandler` 做逐架构分派 |
| [调试](debug/index.md) | 把 IR lower 成可执行的 PyTorch 脚本用于数值校验 |

## 顶层主题

| 页面 | 内容 |
| ---- | ---- |
| [PTO 项目生态](00-ecosystem.md) | 多仓库工具链 —— PyPTO、PTOAS、pto-isa、simpler、pypto-lib —— 及其组合方式 |
| [编译性能剖析](01-compile-profiling.md) | 内建的编译流水线墙钟计时 |
| [错误处理](02-error-handling.md) | `CHECK` 与 `INTERNAL_CHECK`、PyPTO 异常类型、失败信息中的 IR 源码位置 |
| [日志](03-logging.md) | 两套相互独立的日志子系统，以及如何判断一条消息来自哪一套 |
| [运行时 DFX 开关](03-runtime-dfx.md) | 通过 `RunConfig` 暴露的五个运行时诊断子特性 |
| [模拟器 Trace 清洗](04-simulator-trace-cleaning.md) | 把 MindStudio Insight 二进制 dump 转成可读 trace |
| [逐任务 Ring Sizing](05-runtime-ring-sizing.md) | `RunConfig` 上的三个 ring 尺寸覆盖项及其调优时机 |
| [持久化 L3 执行](06-persistent-l3.md) | 在多个已 prepare 的分布式程序间复用同一个 worker |
| [内存图](07-memory-map.md) | 把 pass dump 渲染成可交互的片上内存 HTML 图 |
| [分布式算子](distributed_ops.md) | N6 分布式算子家族 —— 对集合通信与低层原语的类型化 DSL 访问 |
| [PTOAS 算子状态矩阵](ptoas-op-status.md) | 编译器当前会发射哪些 PTOAS 公开与兼容算子 |
| [A5 Mixed MX Scale 传输](design/a5-mixed-mx-scale-transport.md) | V2C 传 scale 失败分析与「scale 强制 GM」决策 |

## 另请参阅

- [PTO ISA 参考](../reference/index.md) —— 后端所面向的硬件模型。
- [运行时文档](https://hw-native-sys.github.io/simpler/) —— 执行已编译程序的调度器。
