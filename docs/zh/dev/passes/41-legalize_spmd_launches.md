# LegalizeSpmdLaunches

文档编号为 **41**，位于 LowerL2TensorCollectives 之后、
DeriveCallDirections 和 AutoDeriveTaskDependencies 之前。计入实用 pass
NormalizeStmtStructure 后，它是默认流水线的第 **42** 个执行项；
文档编号不计入该实用 pass。

零块 SPMD 计算任务可能触发 AICPU 异常。此 pass 使用 `arith::Analyzer`
证明 `core_num >= 1`，包括纯标量 SSA 绑定、循环范围和外层分支约束。
能证明为正的 launch 保持不变；否则生成 `if core_num > 0`，非正路径跳过
计算任务并保留调用方已有的输出 Tensor。DSL parser 仍拒绝非正字面量。

## 输出契约

每个 Tensor Out/InOut 参数都必须具有调用方提供的实参，即使它没有返回或
没有被使用。遗漏时，编译错误列出所有缺失 Tensor 的名称、从零开始的参数
位置、launch 名称和源码位置。应提前分配并传入这些 Tensor。
只有 Out 参数而没有 In 参数的 kernel 同样受支持。

返回值通过 NormalizeReturnOrder 建立的规范返回形式，按 Var 身份映射到
callee 参数，再映射到调用方实参。多输出、重排返回和 view 分别保留对应关系。
无法映射到现有 Tensor 的逃逸返回值会报错。空路径不会初始化输出内容，
后续 consumer 看到的是 buffer 之前的内容。已经证明为正的 launch
仍可使用 runtime 自动分配的输出。

## SSA 与依赖

概念 IR（实际打印的临时变量名称可能不同）：

```python
if n > 0:
    (a_then, b_then), tid_then = pl.spmd_submit(
        self.kernel, a0, b0, core_num=n, deps=[prior]
    )
    a, b, tid = pl.yield_(a_then, b_then, tid_then)
else:
    tid_empty = pl.system.task_dummy(deps=[prior])
    a, b, tid = pl.yield_(a0, b0, tid_empty)
result, _ = pl.submit(self.consumer, a, b, deps=[tid])
```

IfStmt 在两个分支之外仅定义一次合并结果。Tuple 投影在正分支内展开，
consumer 引用外层结果。生成的 Tensor 结果采用对应调用方 buffer 的类型，
使动态 shape 变量属于调用方而非 callee。即使没有计算任务，dummy task 仍保留显式前驱关系。
正分支保留原 launch 的 metadata。

AutoDeriveTaskDependencies 从相同的入口存储历史分别分析两条分支，将 yield
的 TaskId 提升到 If 结果，并保留空路径所需的早期 producer。无法安全提升
局部 producer 时回退到 runtime TensorMap 推导，不导出分支内的依赖变量。
用户 manual scope 继续使用原有显式依赖。

单一 launch 专用的编译器 GM pipe scratch 分配随 launch 移入正分支，
以保持分配大小推导与 dispatch 相邻。用户输出分配保留在 guard 外；
共享编译器 scratch 会报错。Graph 函数内的动态 launch 仍受 Graph 验证限制。

## API 与属性

```python
from pypto import passes
result = passes.legalize_spmd_launches()(program)
```

要求 SSAForm、NoNestedCalls 和 ReturnParamsExplicit；保留 SSAForm、
NoNestedCalls，建立 NormalizedStmtStructure。控制流改写使调用方向、runtime
scope 和循环 carry 分类失效，这些分析必须随后运行。此 pass 具有幂等性，
无需配置开关。遍历和缓存查找的复杂度与 IR 及调用参数总大小呈线性关系。

Codegen 还会再次进行算术检查；如果遗漏此 pass，或后续 pass 引入了没有正数
保护的 launch，会在代码生成前报错。
