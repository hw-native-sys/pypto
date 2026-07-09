# EliminateRedundantVarCopy

对编排（Orchestration）函数中冗余的 `X = Y` 变量重绑定做复制传播（copy propagation）：
把 `X` 的所有使用点改写为 `Y`，并删除该拷贝语句。

- **位置**：第 35 个 pass，紧跟在 [`DeriveCallDirections`](34-derive_call_directions.md) 之后
- **作用范围**：仅 `FunctionType::Orchestration` 函数体
- **属性**：requires `SplitIncoreOrch`、`CallDirectionsResolved`；两者均保留
- **源码**：`src/ir/transforms/eliminate_redundant_var_copy_pass.cpp`

## 动机

经过 outlining、SSA 转换和方向推导之后，编排函数体里会累积一批纯 SSA 重绑定，
其右值只是另一个 `Var`/`IterArg`：

```python
score__ssa_v1: pl.Tensor[[64], pl.FP32] = score__rv_v2   # same physical buffer
```

`X` 和 `Y` 指向**同一块物理 buffer**——编排层的 `Tensor` 是句柄（handle）而非值，
所以这条重绑定什么都没拷贝。把它们留在 IR 里，编排 codegen 就不得不在发射期处理它们：

- 两侧折叠到同一个 param 血缘发射名的重绑定会生成 `auto X = X;`，
  gcc 报错 *"use of 'X' before deduction of 'auto'"*。
- `pl.manual_scope` 内部的重绑定会发射一个块局部的 `Tensor X = Y;`，
  它的名字在右花括号处失效；于是排在 scope **之后**的 task 引用到越界的 `X`，
  生成的 `.cpp` 编译失败（issue [#1697](https://github.com/hw-native-sys/pypto/issues/1697) /
  [#1713](https://github.com/hw-native-sys/pypto/issues/1713)）。

Codegen 用一段发射期的补丁（`FIXME(#1281)`）遮盖了这两点，违反了
[`00-pto_codegen.md`](../codegen/00-pto_codegen.md) 里严格 1-to-1 翻译的契约。
本 pass 在 IR 层消除掉所有能被证明安全的别名，codegen 因此无需再对它们做推理。
补丁仍需覆盖的残余情形见[局限](#局限)。

## 变换

对每条 `AssignStmt` `X = Y`（`Y` 是 `Var`/`IterArg`），**同时满足**以下护栏时折叠
（把 `X` 的所有使用改写为 `Y`，删除该语句）：

| # | 护栏 | 理由 |
| - | ---- | ---- |
| 1 | 所在函数是 `Orchestration` | InCore/Group/Spmd 函数体不会携带这类重绑定；分布式 `host_orch` 有自己的 codegen |
| 2 | `X`、`Y` 都不是 carry 左值 | codegen 把 `iter_args` / `return_vars` 当作跨迭代、跨分支相位重新赋值的 C++ 局部变量；折叠会破坏快照语义 |
| 3 | `X`、`Y` 同一 buffer root | 保证这是纯别名，而不是真实的数据搬运 |
| 4 | `Y` 的定义区域包含 `X` 的每一个读点 | 定义在 `manual_scope` 内的源，在 scope 之后的读者处已经失效；折叠到它身上只是把 #1713 挪个位置，而不是修好 |

Buffer 身份复用 `BufferRootCollector`（`orchestration_analysis.h`），
与 `DeriveCallDirections` 使用的是同一个判据来源。

护栏 4 正是解决 #1713 的关键。一条根植于**外层** base 的 scope 内部重绑定链会
一路折叠到该 base，于是 scope 之后的读者引用的是外层变量，块局部别名根本不会被发射：

```python
# Before
with pl.scope(mode=pl.ScopeMode.MANUAL):
    x__rv_v2 = base          # block-local alias of an outer tensor
    x__rv_v5 = x__rv_v2
r = self.rd(x__rv_v5)        # after-scope reader -> `x__rv_v5` is dead in C++

# After
with pl.scope(mode=pl.ScopeMode.MANUAL):
    pass
r = self.rd(base)            # resolves in the enclosing frame
```

反过来，**定义在 scope 内部**的源会被原样保留，因为折叠并不能让它更可见：

```python
with pl.scope(mode=pl.ScopeMode.MANUAL):
    p__ssa_v0 = pl.tensor.create([64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
    x__ssa_v1 = p__ssa_v0    # kept: `p__ssa_v0` is scope-local
r = self.rd(x__ssa_v1)
```

拷贝链会被传递性地解析：`X = Y; Z = X` 会让 `X` 和 `Z` 的所有使用都指向 `Y`。
共享同一个最终源的候选会被**整组**丢弃，因此一条链绝不会只折叠一半。

不折叠的情况：右值是 `Call`/`Submit`/`MakeTuple`/`TupleGetItemExpr`（这些是真正的定义），
以及 TaskId 标量（没有 buffer root）——它们的别名映射仍由编排 codegen 跟踪。

## 对生成代码的影响

一个自始至终只是 `pl.Out` 参数别名的循环 carry 会连同它的自赋值一起完全消失：

```cpp
// Before
Tensor k_proj_rv = ext_k_proj;
for (...) {
    Tensor k_proj_iter = k_proj_rv.view(...);
    rt_submit_task(...);              // writes the buffer in place (add_inout)
    Tensor k_proj_next = k_proj_rv;
    k_proj_rv = k_proj_next;          // no-op
}

// After
for (...) {
    Tensor k_proj_iter = ext_k_proj.view(...);
    rt_submit_task(...);
}
```

## 局限

护栏 2 拒绝任何触及 carry 左值的拷贝，因此**循环 carry 的循环后重绑定**
（`score__ssa_v1 = score_rv`）会被留在 IR 里。当该循环位于 `pl.manual_scope` 内、
且这条拷贝在 scope 之后被读取时，编排 codegen 仍需要它的 `FIXME(#1281)` 发射期
护栏来重映射名字。

要正确折叠这一情形，必须区分「循环后读取 carry」（carry 已稳定，折叠安全）与
「循环体内、yield 之前取的拷贝」（折叠会别名到 carry 的**后续**值 —— 参见
`test_manual_scope_in_loop_carry_copy_keeps_snapshot`）。codegen 今天靠把 carry 的
`Tensor carry = init;` 声明**外提**出 manual-scope 体来解决，而这个决策本 pass
无法从 IR 中看到。因此要退役这段补丁，就得把那次外提也搬进 IR；在此之前护栏保留，
而本 pass 只是让它折叠掉的那些别名压根不再产生。

## 复杂度

四次线性遍历（buffer root、carry 集合、拷贝候选、作用域/def-use 映射）加一次改写，
配合哈希表查找 —— `O(N log N)`，满足 `pass-complexity.md`。

## 与打印器的交互

本 pass 可能把某个区域（例如函数体只有重绑定的 `manual_scope`）的语句全部折叠掉。
空的区域体仍然必须打印出一个缩进块，因此 `IRPythonPrinter` 会为它输出 `pass`；
否则打印出的 IR 无法重新解析，流水线的 roundtrip 校验会直接拒绝。

## 测试

`tests/ut/ir/transforms/test_eliminate_redundant_var_copy_pass.py` —— 针对每条护栏
做 before/after 结构对比：折叠参数拷贝、SSA 拷贝链、被就地写的源，以及跨 `manual_scope`
的 #1713 外层 base 链；保留循环 carry、scope 局部源、`Call` 右值和 InCore 函数体。

## 参见

- [`34-derive_call_directions.md`](34-derive_call_directions.md) —— 紧邻在前；提供 `arg_directions` 与 `BufferRootCollector` 用法
- [`05-simplify.md`](05-simplify.md) —— 只做标量常量传播，从不重构语句
- [`../codegen/01-orchestration_codegen.md`](../codegen/01-orchestration_codegen.md) —— 本 pass 所维护的 1-to-1 发射契约
