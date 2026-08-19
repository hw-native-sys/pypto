# 执行

把程序变成产物，再把产物跑到设备上。

在这一章之前，讲的都是你写什么；这一章讲的是它之后会怎样：`ir.compile` 与 `JITFunction.compile` 产出一个 `CompiledProgram`，`ChipWorker` 负责派发它。你真正会关心的东西，多半由两个旋钮决定 —— 产物落在哪里，以及运行时被允许在两次 launch 之间复用其中的什么。

| 页面 | 覆盖 |
| ---- | ---- |
| [编译](00-compile.md) | `ir.compile` 及其参数、`JITFunction.compile`、产物目录、pass dump |
| [运行](01-run.md) | `CompiledProgram` 的契约、`ChipWorker`、`DeviceTensor`，以及影响派发的 `RunConfig` 字段 |

## PyPTO 到哪里为止

PyPTO 产出产物，交给 **simpler** 运行时，后者负责调度、任务环与设备生命周期。本章只记录这条边界上 PyPTO 那一侧：你调用的 API、它产出什么、以及运行时会读它的哪些字段。

边界另一侧的机制 —— 任务如何被调度、依赖如何解析、那些环做什么 —— 见 [运行时文档](https://hw-native-sys.github.io/simpler/)。其中你能从这一侧调的部分，见[内存](../performance/05-memory.md)。

## 参见

- [快速上手](../00-getting_started.md) —— 从一个 kernel 到一个结果的最短路径。
- [工具](../tools/index.md) —— 结果不对时该拿什么。
- [性能](../performance/index.md) —— 度量并调优本章所启动的东西。
