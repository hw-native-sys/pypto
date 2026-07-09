# ClassifyIterArgCarry Pass

对 Orchestration 函数中每个 `ForStmt` 的 iter_arg 做分类——**平凡别名**（trivial
alias）还是**重绑定 carry**（rebind carry）——并为 `pl.manual_scope` 内的
`Scalar[TASK_ID]` fence 数组定尺。分类结果以 attr 形式打在 `ForStmt.attrs_` 上，
orchestration codegen 直接读取，不再自行推导。

## 概述

一个编排层循环 carry 在生成的 C++ 中有两种降级方式：

| 分类 | 生成的 C++ | 原因 |
| ---- | ---------- | ---- |
| **trivial（平凡）** | iter_arg 与 return_var 都别名到 init 值的 emit 名 | 运行时依赖追踪器以 `Tensor*` 身份为键，`OUTPUT_EXISTING` / `INOUT` 参数记录的是传入 `Tensor` 左值的地址。为 carry 物化一个新的 `Tensor` 会打断依赖链——kernel 的读写看到的 `&tensor` 与生产者不同。 |
| **rebind（重绑定）** | 声明一个可变 carry 变量，`YieldStmt` 赋值回该变量 | yield 值是*另一个*缓冲区（例如循环体内新建的 tensor）。没有这个 carry，Python 侧的 `current = next` 这类重绑定既不会传递到下一次迭代，也不会传递到循环之后的代码（issue #1286）。 |

当且仅当 iter_arg 的 yield 值落在该 iter_arg 的**别名等价类**（alias class，即指
向同一后端缓冲区的 Var 集合）中时，它才是 trivial。`Scalar[TASK_ID]` carry 永远不
是 trivial：运行时每次迭代都返回一个全新的 `PTO2TaskId`。

在 `pl.manual_scope` 内部，`Scalar[TASK_ID]` 的 rebind carry 还会进一步降级为定长的
`PTO2TaskId[N]` fence 数组。`N` 取自持有该数组的 `pl.parallel` 循环的常量 trip
count；若一个 `Sequential` 循环把该数组穿过内层 `pl.parallel` 向外传递，则继承内层
的 extent。

**何时运行**：`Default` 与 `DebugTileOptimization` 策略的最后一个 pass，紧跟在
[`MaterializeRuntimeScopes`](41-materialize_runtime_scopes.md) 之后。跑在最后意味着
被分类的 IR 与 codegen 实际降级的 IR 完全一致。

## 别名等价类（alias class）

有四条规则把一个 Var 放进某个 iter_arg 的别名类。每条规则只给出*一个*别名来源，因此
这些边构成一片森林，等价类查询退化为带记忆化的链式走查（O(N)），而不是不动点迭代：

| 规则 | 边 |
| ---- | -- |
| `tensor.assemble` | 结果别名到 `args[0]`（写入目标） |
| Out / InOut 调用 | 结果别名到 callee 真正*返回*的那个 Out/InOut 实参（经 `return_lineage` 追踪，因此 GM scratch 的 Out 参数不会误抢别名） |
| `TupleGetItemExpr` | `ret_tuple[i]` 别名到产生该 tuple 的 `Call` / `Submit` 的第 i 个输出侧实参 |
| 嵌套 `ForStmt` | 穿过嵌套循环的 carry 以该循环的 `return_var` 重新出现，而它别名到嵌套循环的 init 值 |

assemble 规则与 Out/InOut 规则不可能在同一条赋值上同时命中：`tensor.assemble` 是
builtin op，而 `DeriveCallDirections` 只给非 builtin 调用打 `arg_directions`。

`ArrayType` 的 iter_arg 被**排除**在嵌套循环规则之外。与 `TensorType`（指向缓冲区的
指针别名）不同，`ArrayType` iter_arg 在每一层都拥有一份*全新的* C 栈数组。把内层
return_var 当作外层 iter_arg 的别名，会把外层 slot 误标为 trivial，从而静默丢掉外层
的 yield 回写拷贝——而这正是 `SEQ x PARALLEL` phase fence 中跨阶段传递状态的机制。

## 打上的属性

`ForStmt::attrs_` 是一个扁平的 `string → 标量` 映射，因此 plan 使用带下标后缀的 key：

| Key | 类型 | 含义 |
| --- | ---- | ---- |
| `iter_arg_rebind_<i>` | `bool` | `True` = 物化 carry，`False` = 平凡别名。**每个** slot 都会打上，因此该 attr 的存在本身就证明 pass 跑过了。 |
| `iter_arg_array_size_<i>` | `int` | `PTO2TaskId[N]` fence 数组的 extent。仅在为正时打上；缺失表示走标量 / tensor / `ArrayType` carry 路径。 |

读取请使用 `ir::transform_utils::IterArgIsRebind()` / `IterArgArraySize()`
（`include/pypto/ir/transforms/utils/transform_utils.h`），不要手写字符串匹配 key。

## 示例

```python
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    for _i, (acc,) in pl.range(0, 4, init_values=(out,)):
        acc2 = self.accumulate(x, acc)   # 就地写 `acc` 并返回它
        (out,) = pl.yield_(acc2)
    return out
```

pass 之后该循环带上 `attrs={"iter_arg_rebind_0": False}`：`acc2` 经 InOut 回写规则
别名到 `acc`，于是 codegen 把 `acc` 和 `out` 都路由到参数的 emit 名，并跳过 yield
自赋值。

把循环体换成 `pl.create_tensor` 的结果，则得到
`attrs={"iter_arg_rebind_0": True}`，codegen 声明 `Tensor <carry> = <init>;` 并在
yield 处赋值。

在 manual scope 内，`pl.parallel(4)` 上的 TaskId carry 得到
`attrs={"iter_arg_rebind_0": True, "iter_arg_array_size_0": 4}`，降级为
`PTO2TaskId arr[4];`。

## 报错

携带 manual-scope 依赖（`deps=[...]`）的 `pl.parallel` 循环必须有静态可知的 trip
count——运行时 fence 需要定长 `N` 的 `PTO2TaskId[N]` 数组。动态 trip count 会抛出面向
用户的错误：

```text
manual_scope: pl.parallel loops carrying a manual_scope dep (via ``deps=[...]``)
must have a statically-known trip count. ...
```

该诊断在本 pass 阶段抛出，早于 codegen。

## Pass 属性

| - | 属性 |
| - | ---- |
| Required | `CallDirectionsResolved`, `RuntimeScopesMaterialized` |
| Produced | `IterArgCarryClassified`, `RuntimeScopesMaterialized` |
| Invalidated | — |

`IterArgCarryClassified` 是 codegen 的前置条件（见
`VerifyOrchestrationCodegenPreconditions`），并注册了对应的 property verifier：一个
带 iter_args 却没有 `iter_arg_rebind_<i>` attr 的 `ForStmt` 说明 pass 没跑，那样
codegen 会把所有 carry 静默降级成平凡别名。

## 编译器推导的依赖 carry

对于收集*编译器推导*任务依赖的 iter_arg（`attrs["compiler_manual_dep_edges"]`，由
`AutoDeriveTaskDependencies` 产生），orchestration codegen 会在本 pass 打的 plan 之上
再叠加两个标志。它们依赖 program 全局的依赖边而非循环自身的结构，因此仍留在 codegen：
carry 被强制置为 `rebind`，尺寸取自外层循环的常量 trip count；当该 trip count 是动态
值时，退化为 `std::vector<PTO2TaskId>` 收集。

## 参见

- [MaterializeRuntimeScopes](41-materialize_runtime_scopes.md) —— 紧邻的前一个 pass
- [Orchestration codegen](../codegen/01-orchestration_codegen.md) —— 本 plan 的消费者
- [Pass manager](00-pass_manager.md)
