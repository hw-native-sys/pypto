# LowerGraphForRuntime Pass

默认流水线的首个 Pass 从 `PassContext` 读取目标运行时。在 `host_build_graph`（HBG）
上，Graph 函数和作用域继续走已有的提取、合法化与录制流程。在
`tensormap_and_ringbuffer`（TMR）上，本 Pass 将其降级为普通任务提交所需的形式。

## TMR 降级

`FunctionType::Graph` 函数转为 `FunctionType::Inline`。已有的 `InlineFunctions` Pass
在各调用点代入实参、重命名局部变量并绑定返回值。`GraphScopeStmt` 在所属函数的原位置
展开其作用域体。变换保留原始程序，因此同一份程序也能用于 HBG 编译。

SSA 转换、任务依赖分析和内存规划处理展开后的编排。片段内的循环、分支、kernel 调用和
显式任务依赖保留在 IR 中。生成的编排在每次执行时逐个提交任务。

Graph 边界的张量 dump 标记沿用 Inline 的传递机制。TMR 编译会拒绝以 Graph 函数为
目标的显式 `Submit`。Graph 调用或作用域上的其他元数据同样触发诊断，其中包括 Graph
作用域的外层依赖和捕获的完成 TaskId。这些依赖和任务号引用的外层调度边界会在 TMR
降级时消除。

HBG 的录制从独立的自动运行时作用域开始。TMR 将 Graph 展开到外层手动作用域会改变任务
依赖跟踪。TMR 因此拒绝手动作用域内的 Graph 调用和区域，包括经由 Inline 辅助函数进入的
Graph。正常调用 Graph 时，Graph 体内声明的手动作用域会保留。

本 Pass 没有前置或产出的 IR 属性。它在 TMR 上可能引入 Inline 函数，因此会使
`InlineFunctionsEliminated` 失效。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::LowerGraphForRuntime()` | `passes.lower_graph_for_runtime()` | Program 级 |

## 另见

- [InlineFunctions](02-inline_functions.md)
- [OutlineGraphScopes](09-outline_graph_scopes.md)
- [LegalizeGraphBoundary](53-legalize_graph_boundary.md)
