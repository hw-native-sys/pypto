# 用户手册

如何编写、编译、运行与调试 PyPTO 程序。

## 目录

| 页面 | 内容 |
| ---- | ---- |
| [入门指南](00-getting_started.md) | 安装、第一个 tensor 程序、tile kernel、循环、多函数 program、编译与运行、`DeviceTensor` 与显式派发 |
| [语言指南](01-language_guide.md) | 类型系统、program 与函数、操作、SSA 与控制流、内存与数据搬运、InCore 作用域、编译 |
| [操作参考](02-operation_reference.md) | `pl.*`、`pl.tensor.*`、`pl.tile.*` 三个命名空间的算子全貌 |
| [Torch Codegen 调试指南](03-torch_codegen_debug.md) | 从 IR 生成 PyTorch 参考实现，用于定位精度问题 |

## 阅读路径

1. **[入门指南](00-getting_started.md)** —— 先端到端跑通一个程序，再读其他内容。
2. **[语言指南](01-language_guide.md)** —— 理解刚才跑通的那段代码背后的语法与语义。
3. **[操作参考](02-operation_reference.md)** —— 写 kernel 时按需查阅。
4. **[Torch Codegen 调试指南](03-torch_codegen_debug.md)** —— 输出与参考实现对不上时使用。

## 尚未收录的内容

本手册正在扩展为完整的分章结构 —— 教程、分布式编程、性能优化、精度定位各自成章。
在这些章节落地之前，相应内容位于[开发者文档](../dev/index.md)：

| 主题 | 当前位置 |
| ---- | -------- |
| 任务与依赖、`manual_scope` / `submit` | [Python IR 语法规范](../dev/language/00-python_syntax.md)、[AutoDeriveTaskDependencies](../dev/passes/36-auto_derive_task_dependencies.md) |
| 分布式 DSL 与集合通信 | [分布式算子](../dev/distributed_ops.md) |
| 性能提示与诊断 | [诊断](../dev/passes/92-diagnostics.md)、[编译性能剖析](../dev/01-compile-profiling.md) |
| 运行时 DFX 开关、ring sizing、memory map | [运行时 DFX](../dev/03-runtime-dfx.md)、[逐任务 Ring Sizing](../dev/05-runtime-ring-sizing.md)、[内存图](../dev/07-memory-map.md) |
| 外部 C++ kernel | [集成手写 C++ Kernel](../dev/language/01-external-kernels.md) |

## 另请参阅

- [开发者文档](../dev/index.md) —— 编译器如何 lower 你写下的代码。
- [PTO ISA 参考](../reference/index.md) —— 生成代码背后的指令语义。
- [运行时文档](https://hw-native-sys.github.io/simpler/) —— 执行已编译程序的调度器。
