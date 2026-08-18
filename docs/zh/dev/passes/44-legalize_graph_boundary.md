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

## Step B —— 边界张量的派生切片

回放 patch 的是边界张量的**地址**。在区域**内部**取的 view 会从录制时冻结下来的
东西重新推导，所以必须改到调用点去取：

```python
wl = pl.tensor.slice(w, [128, 128], [layer_idx * 128, 0])   # 在区域内部
```

Step B 把这个切片搬出去，把结果作为一个新的边界张量传进来。每个切片点各自成为一个
形参、各自带固定形状 —— 这正是 runtime 的 `BOUNDARY_VIEW` 分类所要求的：它按
「同 buffer + 偏移」匹配，形状根本不参与，所以一个形状逐次变化的 view 压根无法被
分类。

外提出来的语句按**先标量、后张量**发射，因为切片的偏移通常就是 Step A 的标量，绑定
必须先于使用。而**形参**顺序恰好相反 —— 张量在前、标量在后 —— 这是 `CoreTaskArgs`
的要求。只有对边界**形参**的切片会被外提；对区域局部张量取的 view 保持原样。

## Step C —— 区域内的分配

codegen 会把 `pl.create_tensor` 降级成批量 `alloc_tensors`，而被录制区域内任何一个
裸的 `alloc_tensors` 都会让 runtime 判定该录制 unsupported。本 pass 对此**报错**，
而不是把分配外提：

> Graph function 'layer' allocates a tensor inside the region. […] Allocate it in
> the caller and pass it in as a `pl.InOut` parameter instead.

自动外提会新增第二个 `InOut` 形参，而返回值别名映射要求被调方的 `ReturnStmt` 直接
指名某个形参，才能确定张量返回值别名到哪一个 —— 合成出来的形参不满足这个不变量。
要自动化，得先把那套映射改造掉。

注意只有**裸的** `alloc_tensors` 会毒化录制；逐任务的 `add_output(TensorCreateInfo)`
是合法的，runtime 自己的 graph 系统测试就是这么分配的。

## Step D —— 边界合法性

| 检查 | 原因 |
| ---- | ---- |
| 至少 1 个张量形参 | 空边界的 graph 回放时无处可 patch，runtime 拒绝缓存 |
| 至多 32 个张量形参 | runtime 的边界上限 |
| 不允许 `Out` 张量形参 | `Out` 意味着 runtime 分配该 buffer；被录制的 graph 其边界张量必须已存在，回放才能 patch 地址 |
| 标量形参必须是 `In` | 边界标量按值传入、由调用点回放 |
| 只能返回自己的形参 | `rt_submit_graph` 只在缓存**命中**时才返回有效 task id，所以任何东西都不能依赖 graph 调用的结果。`return c`（`c` 为 `InOut` 形参）是原地写的写法，可以；返回计算出的新值不行 |
| 至多 1024 个被拉起的任务 | runtime 的单图节点上限 |
| Graph 不能调用 Graph | runtime 无法在正在录制的 graph 内部再录一个 graph |
| 调用点必须传满全部形参 | `Submit` 通常允许只传前缀、由 runtime 分配尾部 `Out` 形参；Graph 没有这种尾部 |
| launch 上不能带显式依赖 | 显式依赖边会让该次 launch 不可缓存，区域就会静默退化成普通任务 |
| launch 上不能带 dispatch predicate | graph launch 上的 predicate 既不被遵守也不被拒绝 —— runtime 静默清零它，于是区域会无条件执行 |

## 在流水线中的位置

跑在最后一个 `Simplify` 之后、
[`MaterializeRuntimeScopes`](45-materialize_runtime_scopes.md) 之前。

这个位置是两边夹出来的。`DeriveCallDirections` 和 `AutoDeriveTaskDependencies`
必须已经跑完，这样实参方向与跨任务边才是已知的；而 `MaterializeRuntimeScopes`
必须还没跑，这样 Step A 要搬动的语句外面还没有被套上 scope。

## Pass 属性

- **requires**：`SplitIncoreOrch`、`CallDirectionsResolved`
- **produces**：`GraphBoundaryLegalized`、`CallDirectionsResolved`

之所以重新声明 `CallDirectionsResolved`，是因为本 pass 改写了调用实参及其方向
attr；紧随其后的 `MaterializeRuntimeScopes` 要求该属性。

## 尚未处理

把区域内的局部分配自动外提为边界形参（见 Step C），以及把超过 32 个张量的边界
自动打包进 scratch arena。

## 另见

- [Pass Manager](00-pass_manager.md) —— 完整流水线顺序
- [MaterializeRuntimeScopes](45-materialize_runtime_scopes.md) —— 紧随其后运行
