# LegalizeGraphBoundary Pass

让每个 `FunctionType::Graph` 函数都能被 `host_build_graph` runtime 合法地录制与
回放：把 Graph 函数体内派生出来的边界标量外提到调用点，并拒绝那些 runtime 不会
缓存的边界形态。

## 概述

`host_build_graph` runtime 在第一次调用时录制 Graph 函数的任务拓扑，之后回放这份
录制。回放只 patch 两样东西：边界张量的地址，以及边界标量的值。其余一切 —— 节点
数、形状、依赖边、block 数 —— 都被烙进录制下来的 Definition。

由此产生两类问题，而且在 runtime 侧都是静默的：

| 问题 | runtime 的行为 | 本 pass 的处理 |
| ---- | -------------- | -------------- |
| 边界标量在区域内被**派生**出来 | 归类为静态数据，把第一次调用的值冻进录制。永远不告警。 | **Step A** —— 把计算外提到调用点 |
| 边界本身不可缓存 | 拒绝缓存，静默地按普通任务执行该区域 | **Step D** —— 编译期拒绝 |

前者产生错误结果。后者结果正确但完全没有预期的加速 —— 任何数值测试都看不见，这
正是这些检查放在这里、而不是交给一条 runtime 日志的原因。

## Step A —— 派生的边界标量

边界标量是靠**指针身份**追踪的。录制时 runtime 锚定每个 `args.scalar(k)` 槽位的
地址，回放时重新读这些地址。函数体自己算出来的值没有槽位：

```python
@pl.function(type=pl.FunctionType.Graph)
def layer(self, cur, wq, layer_idx: pl.Scalar[pl.INDEX]):
    base = layer_idx * 5120          # <- 派生值：没有实参槽位
    ...                              #    被冻结在第一次调用的取值上
```

Step A 把它改写成以形参形式传入：

```python
# pass 之后，概念上是：
def layer(self, cur, wq, layer_idx, base):   # base 成为真正的边界标量
    ...

# 每个调用点：
self.layer(cur, wq_view(i), i, i * 5120)     # 算术搬到了这里
```

一个值可外提的条件是：它整棵表达式树的叶子只有该 Graph 自己的标量形参和常量 ——
这恰好是调用点能够重算的集合，因为调用点本来就提供这些形参。PyPTO 里的标量算术
是 `BinaryExpr` / `UnaryExpr` 节点而非 `Call`，所以判定沿这两个基类递归，其余节点
一律当作叶子。

如果一个标量流入了任务却**不可**外提 —— 因为它依赖任务输出、张量读取或运行时查询
—— 就会报错，消息里点名该变量并说明为什么这个值无法在调用点重建。

新形参是**追加**而不是前置的：`CoreTaskArgs` 要求所有张量实参排在所有标量实参之前。

## Step D —— 边界合法性

| 检查 | 原因 |
| ---- | ---- |
| 至少 1 个张量形参 | 空边界的 graph 回放时无处可 patch，runtime 拒绝缓存 |
| 至多 128 个张量形参 | `GRAPH_MAX_TENSOR_ARGS` —— 边界是定长的 `GraphTaskArgs` |
| 至多 64 个标量形参 | `GRAPH_MAX_SCALAR_ARGS`。在 Step A 之后检查：Step A 会**新增**标量形参，所以上提前放得下的签名，上提后可能放不下 |
| 不允许 `Out` 张量形参 | `Out` 意味着 runtime 分配该 buffer；被录制的 graph 其边界张量必须已存在，回放才能 patch 地址 |
| 标量形参必须是 `In` | 边界标量按值传入、由调用点回放 |
| 只能返回自己的形参 | `rt_submit_graph` 只在缓存**命中**时才返回有效 task id，所以任何东西都不能依赖 graph 调用的结果。`return c`（`c` 为 `InOut` 形参）是原地写的写法，可以；返回计算出的新值不行 |
| 被拉起的任务数在 1..1024 之间 | `graph_execution_storage_layout` 既拒绝 0 个节点，也拒绝超过 `GRAPH_MAX_NODES` 的。循环内的 launch 按迭代次数计入而非按词法调用点计 1；`system.task_dummy` 也计入——它 lower 成 `rt_submit_dummy_task`，且 `ExpandManualPhaseFence` 会自动插入。分配同样会记节点，按**上界**计——通过这项检查即意味着 runtime 会接受该 Graph。codegen 会收集一个语句列表里所有符合条件的 create（中间夹着 launch 不会打断批次），再按每次 `alloc_tensors` 最多 `kAllocTensorsArgs`（16）个打包。它的三条不合格规则里有两条在这里不可能触发（shape 读局部变量已被按非常量拒绝；SSA 下不可能出现已声明的 var），这些 create 是**精确**计数的。第三条可能触发——被注入的 GM pipe buffer 在其 `core_num` 读到 body-local 时会离开共享批次，而这只有 emitter 的 use-resolution 才知道——所以这类按最坏情况各计 1 个节点。批量大小和 GM-pipe 判定与 emitter 共用 `utils/alloc_batching.h`，而非各自重述 |
| 运行时循环 / 分支内不得有分配 | 每次分配都记一个节点，所以数量随调用变化就是拓扑随调用变化 |
| 分配的 shape 必须是编译期常量 | 录制会把 shape 抄进节点并据此推出缓冲区地址；读取边界标量的 shape 会被冻结在首次调用的值上 |
| 区域内不得有 `tensor.full` | orchestration codegen 没有它的 lowering，会按 misplaced tensor op 拒绝 |
| 运行时循环 / `while` / `if` 内不得有 launch | 录制在首次调用时定死拓扑并原样回放，因此随调用变化的 launch 次数或分支会静默重放第一次的形状 |
| 任务实参里不得内联计算标量 | Step A 只上提**具名**的派生值；写在调用处的内联表达式没有名字可提、也没有边界 slot，会被冻结在首次调用的值上 |
| Graph 不能调用 Graph | runtime 无法在正在录制的 graph 内部再录一个 graph |
| 调用点必须传满全部形参 | `Submit` 通常允许只传前缀、由 runtime 分配尾部 `Out` 形参；Graph 没有这种尾部 |
| launch 上不能带显式依赖 | 显式依赖边会让该次 launch 不可缓存，区域就会静默退化成普通任务 |
| launch 上不能带 dispatch predicate | graph launch 上的 predicate 既不被遵守也不被拒绝 —— runtime 静默清零它，于是区域会无条件执行 |

## 在流水线中的位置

跑在最后一个 `Simplify` 之后、
[`MaterializeRuntimeScopes`](46-materialize_runtime_scopes.md) 之前。

这个位置是两边夹出来的。`DeriveCallDirections` 和 `AutoDeriveTaskDependencies`
必须已经跑完，这样实参方向与跨任务边才是已知的；而 `MaterializeRuntimeScopes`
必须还没跑，这样 Step A 要搬动的语句外面还没有被套上 scope。

## Pass 属性

- **requires**：`SplitIncoreOrch`、`CallDirectionsResolved`
- **produces**：`GraphBoundaryLegalized`、`CallDirectionsResolved`

之所以重新声明 `CallDirectionsResolved`，是因为本 pass 改写了调用实参及其方向
attr；紧随其后的 `MaterializeRuntimeScopes` 要求该属性。

## 尚未处理

对边界张量的派生**切片**另行处理。区域内的分配在这里是允许的，受上面的常量
shape 规则约束；把它外提到调用点是后续的改动。

## 另见

- [Pass Manager](00-pass_manager.md) —— 完整流水线顺序
- [MaterializeRuntimeScopes](46-materialize_runtime_scopes.md) —— 紧随其后运行
