# HoistScopeLocalAllocs Pass（分配外提）

给每个**直接**位于 `pl.manual_scope` 体内、且 *enclosing-scope-valid*（外层作用域
可见）的 `tensor.create` 打上 `hoistable_alloc` 属性，使 orchestration codegen 把
该 buffer 的声明外提一层 C++ 作用域，而不再靠 emit 期缩进算术恢复外提集合
（issue #1697）。

## 概览

`pl.manual_scope` 会降级为一个 `PTO2_SCOPE(PTO2ScopeMode::MANUAL) { ... }` C++ 块。
manual scope 是**调度**（scheduling）区域，而非**存储**（storage）区域：块内分配的
buffer 可能被排在块 *之后* 的 task 读取。若该 buffer 的声明留在块内，则这个 C++ 局部
变量会在闭括号处出作用域，块后 reader 引用它就会编译失败。

`alloc_tensors(...)` 没有任何调度依赖，所以把它外提一层（放到
`PTO2_SCOPE(MANUAL) {` 头之前）在语义上是等价（inert）的，并且能让该 buffer 在块内
与所有块后 reader 处都保持在作用域内。

本 pass 把「哪些 create 会被外提」变成显式的 IR 事实。此前 orchestration codegen 在
emit 期靠两个信号恢复它：`scope_hoist_sink_` 指针（当前是否正在缓冲一个 manual
scope？）与缩进比较（`IsAtManualScopeBodyIndent()`——当前是否是该 scope 的*直接*体？），
外加即时的 `ShapeDependsOnLocalVars` 分析。这套缩进启发式与 shape 分析，正是
[「严格 1-to-1 codegen」](../codegen/01-orchestration_codegen.md) 契约要求搬进 pass 的内容。

**使用时机**：在
[`MaterializeRuntimeScopes`](41-materialize_runtime_scopes.md) 之后运行，使 manual-scope
边界是一条显式的 `RuntimeScopeStmt(manual=True)` 边，而非某个缩进层级。仅处理
`FunctionType::Orchestration` 函数。

## 标记条件

一个 `tensor.create` `Call` 会被打上 `hoistable_alloc = True`，当且仅当：

1. 它的 `AssignStmt` 是某个 `RuntimeScopeStmt(manual=True)` 体的**直接**子语句
   （嵌套在该 scope *内部* 某个 for/if 里的 create 保持原地——它属于 loop/branch 的
   C++ 作用域，而不是 manual 块）。
2. 它的结果 shape **不**引用该 manual-scope 体内定义的任何 `Var`——即该 buffer 是
   *enclosing-scope-valid*（外层可见）。若某个 shape 维度依赖 scope-local 值，则外提
   到外层后无法求值，因此这种 create 保持原地。

支持嵌套 manual scope：每个 scope 只标记它自己的直接体 create，并按各自的体内定义
集合计算，因此从内层 scope 外提的 buffer 会落在外层 scope 的体内。

本 pass 从不改写结构——它只在 create `Call` 上追加属性。物理搬移（把
`alloc_tensors` 批次路由到外层 scope，以及重映射别名到该 buffer 的 kernel 输出）仍留在
codegen；本 pass 只提供决策。

## 打上的属性

| 键 | 类型 | 含义 |
| -- | ---- | ---- |
| `hoistable_alloc` | `bool` | `True` 打在 `tensor.create` `Call` 上 = 把该 buffer 的声明外提出其所在的 `pl.manual_scope`。缺省 = 保持原地。 |

该键可经 printer/parser 往返（见 `python_printer` 的 op-attr 白名单与
`ast_parser._parse_op_attrs`）。

## 示例

```python
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    with pl.manual_scope():
        scratch = pl.create_tensor([64], dtype=pl.FP32)  # 可外提
        a, a_tid = pl.submit(self.k, x, scratch)
    b, _ = pl.submit(self.k, a, x)                        # 在 scope 之后读 `a`
    return b
```

pass 运行后 `scratch` create 带上 `attrs={"hoistable_alloc": True}`。codegen 会把它的
`alloc_tensors` 批次 emit 在 `PTO2_SCOPE(MANUAL) {` 头 *之前*，使
`const Tensor& scratch = ...;` 声明对块后 reader 仍然存活。

嵌套在 scope 内某个 `for` 里的 create、或 shape 引用了 scope 内计算值的 create，都**不会**
被标记，保持在块内。

## Pass 属性

| - | 属性 |
| - | ---- |
| Required | `CallDirectionsResolved`、`RuntimeScopesMaterialized` |
| Produced | `HoistableAllocsMarked`、`RuntimeScopesMaterialized` |
| Invalidated | — |

`HoistableAllocsMarked` 是 codegen 前置条件（见
`VerifyOrchestrationCodegenPreconditions`），并注册了 property verifier：manual-scope 体内
一个 enclosing-scope-valid 的 `tensor.create` 若没有 `hoistable_alloc` 属性，就意味着本
pass 从未运行，codegen 会把该 buffer 留在块内声明——块后 reader 便会引用一个出作用域的
C++ 局部变量（#1697）。

## 参见

- [MaterializeRuntimeScopes](41-materialize_runtime_scopes.md)——物化本 pass 所依赖的 `RuntimeScopeStmt(manual=True)` 边
- [ClassifyIterArgCarry](42-classify_iter_arg_carry.md)——紧邻其前运行的同类属性标记 pass
- [Orchestration codegen](../codegen/01-orchestration_codegen.md)——标记属性的消费者
- [Pass manager](00-pass_manager.md)
