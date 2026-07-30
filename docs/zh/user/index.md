# 用户手册

如何编写、编译、运行与调试 PyPTO 程序。

## 阅读路径

按你当前要做的事挑一条。三条路径都假设[安装](01-installation.md)已完成。

### 我要写第一个 kernel

[快速上手](02-quickstart.md) → [编程模型](03-programming-model.md) →
[语言指南](01-language_guide.md)

先让一个程序编译通过，再理解它在做什么，最后补全语言表面。过程中把
[操作参考](02-operation_reference.md) 开在旁边 —— 起步阶段你会不停查算子。

### 我有 kernel，但数值不对

[Torch Codegen 调试指南](03-torch_codegen_debug.md) →
[编程模型 § 执行模型](03-programming-model.md)

把 IR lower 成 PyTorch 脚本，逐张量对拍。如果结果是**每次运行都不一样**而不是稳定地错，
问题在顺序而不在算术 —— 去读执行模型那一节，因为语句顺序并不约束执行顺序。

### 我有 kernel，但太慢

[编程模型 § 内存层次](03-programming-model.md) →
[诊断](../dev/passes/92-diagnostics.md) →
[运行时 DFX](../dev/03-runtime-dfx.md)

在动手测量之前，先看编译产物里的 `report/perf_hints.log` —— 编译器可能已经告诉你了。
性能专章尚未编写，其内容当前的位置见下表。

## 目录

| 页面 | 内容 |
| ---- | ---- |
| [安装](01-installation.md) | 前置条件、源码安装、构建选项、验证、`examples/` 导览 |
| [快速上手](02-quickstart.md) | 用 `@pl.jit` 写张量级 kernel（无手工数据搬运）、循环、把工作拆到多个函数、编译与读 IR |
| [编程模型](03-programming-model.md) | 张量 / Tile / Block 三层、控制面与执行面、pass 流水线、内存层次、执行模型 |
| [语言指南](01-language_guide.md) | 完整语言：类型系统、操作、SSA 与控制流、内存与数据搬运、InCore 作用域、编译 |
| [操作参考](02-operation_reference.md) | `pl.*`、`pl.tensor.*`、`pl.tile.*` 三个命名空间的算子全貌 |
| [在设备上运行](00-getting_started.md) | 常驻设备张量、显式派发、性能基准、分布式执行 |
| [Torch Codegen 调试指南](03-torch_codegen_debug.md) | 从 IR 生成 PyTorch 参考实现，用于定位精度问题 |

## PyPTO 提供了什么

| 能力 | 文档位置 |
| ---- | -------- |
| 用 `@pl.jit` 写 kernel（及其所特化成的 `@pl.function` / `@pl.program` 形态） | [快速上手](02-quickstart.md)、[语言指南](01-language_guide.md) |
| 显式片上内存放置（Vec / Mat / L0A / L0B / L0C） | [编程模型](03-programming-model.md) |
| 控制流：循环、携带值、条件、while | [语言指南](01-language_guide.md) |
| 多函数 program 与跨函数调用 | [快速上手](02-quickstart.md) |
| `@pl.jit` 全家族（`.incore`、`.inline`、`.opaque`、`.host`） | [快速上手](02-quickstart.md)、[语言指南](01-language_guide.md) |
| 手写 C++ kernel 接入 | [外部 Kernel](../dev/language/01-external-kernels.md) |
| 设备常驻张量、显式派发、性能基准 | [在设备上运行](00-getting_started.md) |
| 分布式（多卡）程序与集合通信 | [分布式算子](../dev/distributed_ops.md) |
| 对照 PyTorch 参考实现做精度定位 | [Torch Codegen 调试指南](03-torch_codegen_debug.md) |
| 编译期诊断与性能提示 | [诊断](../dev/passes/92-diagnostics.md) |
| 运行时 DFX：swimlane、PMU、依赖图、scope stats | [运行时 DFX](../dev/03-runtime-dfx.md) |
| 片上内存图可视化 | [内存图](../dev/07-memory-map.md) |

## 尚未收录的内容

本手册正在扩展为完整的分章结构 —— 教程、分布式编程、性能优化、精度定位各自成章。
在这些章节落地之前，相应内容位于[开发者文档](../dev/index.md)：

| 主题 | 当前位置 |
| ---- | -------- |
| 任务与依赖、`manual_scope` / `submit` | [Python IR 语法规范](../dev/language/00-python_syntax.md)、[AutoDeriveTaskDependencies](../dev/passes/36-auto_derive_task_dependencies.md) |
| 混合 kernel（AIC + AIV 同一函数） | [LowerAutoVectorSplit](../dev/passes/19-lower_auto_vector_split.md)、[ExpandMixedKernel](../dev/passes/20-expand_mixed_kernel.md)、[TPUSH/TPOP](../reference/pto-isa/01-tpush_tpop.md) |
| 分布式 DSL 与集合通信 | [分布式算子](../dev/distributed_ops.md) |
| 性能提示与诊断 | [诊断](../dev/passes/92-diagnostics.md)、[编译性能剖析](../dev/01-compile-profiling.md) |
| 运行时 DFX 开关、ring sizing、memory map | [运行时 DFX](../dev/03-runtime-dfx.md)、[逐任务 Ring Sizing](../dev/05-runtime-ring-sizing.md)、[内存图](../dev/07-memory-map.md) |
| 外部 C++ kernel | [集成手写 C++ Kernel](../dev/language/01-external-kernels.md) |

## 另请参阅

- [开发者文档](../dev/index.md) —— 编译器如何 lower 你写下的代码。
- [PTO ISA 参考](../reference/index.md) —— 生成代码背后的指令语义。
- [运行时文档](https://hw-native-sys.github.io/simpler/) —— 执行已编译程序的调度器。
