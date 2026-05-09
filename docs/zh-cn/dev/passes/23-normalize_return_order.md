# NormalizeReturnOrder Pass

将每个 InCore 函数的返回 tuple 重新排序，使 `return[i]` 对应声明顺序中的第
i 个 `Out`/`InOut` 参数，并相应地重映射非 InCore 调用方中的
`TupleGetItemExpr` 索引。该 Pass 完成后，编排（orchestration）代码生成可以通
过位置直接映射 `out_indices[i]`，无需追踪 `tile.store` / `ForStmt` yield
链。

## 概述

用户代码可以以任意顺序写 `tile.store` —— 先 `out_b` 后 `out_a`，或者与计算
混排。流水线前期会原样保留 body 顺序，因此 InCore 的 `ReturnStmt::value_`
中各输出可能与声明的 `Out`/`InOut` 参数顺序不一致。若不规范化，编排代码生
成就必须沿着每个 `return[i]` 反向追踪赋值与 `tile.store`，才能确定它实际写入
哪个参数 —— 这种分析应当属于 Pass，而不是代码生成层（参见
`docs/zh-cn/dev/codegen/00-pto_codegen.md`）。

本 Pass 把契约规范化为「按位置 `return[k] ↔ out_indices[k]`」，分两步进行：

1. **Step A（InCore 函数重写）** —— 对每个 `InCore` 函数，计算一个使
   `ReturnStmt::value_` 与声明的 `Out`/`InOut` 参数顺序一致的置换，然后同步
   重写返回值与 `Function::return_types_`。
2. **Step B（调用端索引重映射）** —— 对每个非 InCore 函数（Orchestration /
   Group / Spmd / opaque），重写所有 `TupleGetItemExpr.index_`，前提是它的
   tuple 操作数来源于 Step A 中被重排序的函数调用结果。新索引为
   `permutation[old_index]`，因此调用结果上的观察者仍然把同名 SSA 变量绑
   定到同一物理输出。

对于返回顺序已经与 `Out`/`InOut` 参数声明顺序匹配的函数，本 Pass 是
**no-op**；对于不含 InCore 函数的程序也是 no-op。

**流水线位置**: `Default` 策略中 #20 —— 位于 `SplitVectorKernel`（#19）之
后、`LowerPipelineLoops`（#21）之前。这样既保证所有 kernel 拆分 / tile 结构
决策仍基于原始返回顺序完成，又确保下游 tile 级 Pass（`LowerPipelineLoops`、
`CanonicalizeIOOrder`、`InitMemRef`、`MemoryReuse`、`AllocateMemoryAddr`）
以及最终的 PTO 编排代码生成都能看到规范化后的顺序。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::NormalizeReturnOrder()` | `passes.normalize_return_order()` | 程序级 |

```python
from pypto import passes
result = passes.normalize_return_order()(program)
```

## Pass 属性

| 属性 | 取值 |
| ---- | ---- |
| 前置（Required） | `SplitIncoreOrch`、`IncoreTileOps` |
| 产出（Produced） | — |
| 失效（Invalidated） | — |

`SplitIncoreOrch` 保证 InCore 工作已经被外提为独立函数；`IncoreTileOps` 保
证函数体使用 tile 操作，从而 Step A 所依赖的
`tile.store(_, _, out_param)` 信号一定存在。本 Pass 不产出新属性、也不使
任何属性失效 —— 它保留 SSA、规范化语句结构、内存推断等所有上游已建立的属
性。

## 算法

### Step A —— 计算并应用每个函数的置换

对每个 `InCore` 函数，`BuildReturnToParamMapping` 单次遍历函数体（不含末
尾的 `ReturnStmt`），通过三条规则维护一个 `Var* → out_param_index` 的映
射：

| 规则 | 语句模式 | 行为 |
| ---- | -------- | ---- |
| 1. `tile.store` 写入 Out/InOut buffer | `lhs = tile.store(tile, offsets, out_param, ...)` | `lhs → param_index_of(out_param)` |
| 2. Var 到 Var 别名传播 | `lhs = rhs_var`（且 `rhs_var` 已被映射） | `lhs → lookup(rhs_var)` |
| 3. `ForStmt` iter-arg yield | `for_stmt.iter_args[i].initValue_` 已被映射 | `for_stmt.return_vars_[i] → lookup(initValue)` |

随后对 `ReturnStmt::value_` 中的每个值，先在该映射里查找其 `Var`，否则回
退到与 `Function::params_` 的直接身份匹配；若都未命中则对应位置返回
`kNoParam`，表示该位置「未检测到与 out 参数的关联」，保留其原始下标。

`ComputeReturnPermutation` 把映射变为 `permutation[old_index] = new_index`，
其中 `new_index` 是匹配参数在 `CollectOutIndices(func)` 中的位置。出现以下
任一情况都返回空置换（对该函数即为 no-op）：

- 函数体不含 `ReturnStmt`（开放 IR），或不含任何 Out/InOut 参数。
- `out_indices.size() > ret_to_param.size()` —— 声明的输出参数数量多于返回
  值数量，分析不完整，不能构造越界置换。
- 计算出的置换是恒等置换（已规范）。

当置换非空时，`ReorderReturns` 通过 `MutableCopy` 克隆出新的 `Function`，
将末尾的 `ReturnStmt` 替换为
`value_[permutation[i]] = old_value_[i]` 的版本，并同步置换
`Function::return_types_`，使类型列表与值列表始终对齐。

### Step B —— 重映射调用端的 `TupleGetItemExpr`

对（Step A 已重写后的）程序中的每个非 InCore 函数，
`TupleIndexPermutationMutator` 单次 SSA 遍历做两件事：

- 对每个 RHS 是 `Call(GlobalVar)`、且被调函数在 Step A 中被重排序的
  `AssignStmt`，把 `assign.var → permutation_ref` 记入
  `reordered_tuple_vars_` 映射。
- 一旦该 `Var` 被重新赋值（赋给非重排序函数的调用、非 Call 表达式等），立即
  移除其条目，避免基于身份的查找读到失效绑定。
- 对每个 `TupleGetItemExpr(tuple_var, k)`，若 `tuple_var` 在该映射中，把索
  引重写为 `permutation[k]`。

由于 Step A 重写函数签名与 Step B 重写调用端索引在同一次 Pass 调用中完成，
出口程序状态自洽：每个 tuple 元素仍然绑定到同一个物理输出 buffer，只是用
新的下标访问。

## 约束

| 约束 | 原因 |
| ---- | ---- |
| Step A 仅重写 `InCore` 函数 | 其他函数类型（`Orchestration` / `Group` / `Spmd` / opaque）遵循用户声明的返回形态；它们的调用端在 Step B 中被重映射，但函数体本身不动 |
| `out_indices.size() > ret_to_param.size()` 时跳过 | 不完整分析不能产生越界置换 —— 保留原状，让 verifier 捕获不一致 |
| 恒等置换 ⇒ 不重写 | 避免不必要的 `Function` 克隆，使 Pass 幂等 |
| Step B 仅改写 `VisitExpr` 后 tuple 操作数仍为已记录 `Var` 的 `TupleGetItemExpr` | Mutator 保留 `Var` 节点身份，因此操作数指针在 `reordered_tuple_vars_` 中仍是有效的查找键；即便未来某次改写返回新节点，查找 post-visit 的指针也能保证正确性 |

## 示例

两个 `Out` 参数，但 InCore body 写出顺序与参数声明顺序相反；编排函数按
`ret[0]` / `ret[1]` 默认对应 `out_a` / `out_b` 取出。Pass 完成后，InCore 返
回顺序匹配参数声明顺序，编排函数中的 `TupleGetItemExpr` 下标也被相应重
映射，使同一 SSA 值仍流入 `a` 和 `b`。

**Before**:

```python
@pl.program
class Module:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[16], pl.FP32],
               out_a: pl.Out[pl.Tensor[[16], pl.FP32]],
               out_b: pl.Out[pl.Tensor[[16], pl.FP32]]) \
            -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
        x_tile = pl.load(x, [0], [16])
        a_tile = pl.tile.add(x_tile, x_tile)
        b_tile = pl.tile.mul(x_tile, x_tile)
        out_b_store = pl.store(b_tile, [0], out_b)
        out_a_store = pl.store(a_tile, [0], out_a)
        return (out_b_store, out_a_store)        # ← 与 (out_a, out_b) 声明顺序不一致

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, out_a, out_b):
        ret = self.kernel(x, out_a, out_b)
        a = ret[0]                                # ← 当前实际取的是 out_b
        b = ret[1]                                # ← 当前实际取的是 out_a
        return (a, b)
```

**After**:

```python
@pl.program
class Module:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x, out_a, out_b):
        x_tile = pl.load(x, [0], [16])
        a_tile = pl.tile.add(x_tile, x_tile)
        b_tile = pl.tile.mul(x_tile, x_tile)
        out_b_store = pl.store(b_tile, [0], out_b)
        out_a_store = pl.store(a_tile, [0], out_a)
        return (out_a_store, out_b_store)        # ReorderReturns: 置换 [1, 0]

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, out_a, out_b):
        ret = self.kernel(x, out_a, out_b)
        a = ret[1]                                # TupleIndexPermutationMutator: 0 → 1
        b = ret[0]                                # TupleIndexPermutationMutator: 1 → 0
        return (a, b)
```

同一条 SSA 赋值（`a = ...`）仍然绑定到 `pl.store(a_tile, ..., out_a)` 的输
出；变化的只是 tuple 访问路径。`InOut` 参数的处理方式相同。

完整用例参见
`tests/ut/ir/transforms/test_normalize_return_order.py`：

- `test_swapped_returns_reordered` —— 上文展示的两个 Out 参数案例
- `test_already_ordered_noop` —— 已规范的 IR 保持不变
- `test_single_return_noop` —— 单个 Out 参数无需置换
- `test_non_incore_unchanged` —— 不含 InCore 函数的程序为 no-op
- `test_three_returns_scrambled` —— 三元置换
- `test_2d_tensor_reorder` —— 2 维 tensor / 多维 offset
- `test_inout_param_reorder` —— `InOut` 参数同样参与重排

## 实现

**头文件**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass NormalizeReturnOrder();
```

**实现文件**: `src/ir/transforms/normalize_return_order_pass.cpp`

- `BuildReturnToParamMapping` —— Step A 分析：遍历函数体，将每个
  `ReturnStmt` 值反向映射到 Out/InOut 参数下标。
- `CollectOutIndices` —— 收集 `ParamDirection` 为 `Out` / `InOut` 的参数
  位置。
- `ComputeReturnPermutation` —— 综合上述两个分析，得到最终的
  `permutation[old_index] = new_index`；不需重写或分析不完整时返回空。
- `ReorderReturns` —— 基于 `MutableCopy(func)` 构造新的 `Function`，置换
  `ReturnStmt::value_` 与 `Function::return_types_`。
- `TupleIndexPermutationMutator` —— Step B 改写器：跟踪调用结果变量，并
  重写 `TupleGetItemExpr` 索引。

**属性**: `include/pypto/ir/transforms/pass_properties.h`

```cpp
inline const PassProperties kNormalizeReturnOrderProperties{
    .required = {IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps}};
```

**Python 绑定**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("normalize_return_order", &pass::NormalizeReturnOrder,
           "Create a return order normalization pass\n\n"
           "Reorders return tuple values in InCore functions so that return[i]\n"
           "corresponds to the i-th Out/InOut parameter in declaration order,\n"
           "and updates TupleGetItemExpr indices at call sites accordingly.");
```

**类型存根**: `python/pypto/pypto_core/passes.pyi`

```python
def normalize_return_order() -> Pass:
    """Create a return order normalization pass."""
```

**测试**: `tests/ut/ir/transforms/test_normalize_return_order.py`

## 相关

- [`OutlineInCoreScopes`](10-outline_incore_scopes.md) —— 上游产出本 Pass
  改写的 `InCore` 函数
- [`LowerPipelineLoops`](24-lower_pipeline_loops.md) —— 紧随其后运行；展开
  流水线作用域时消费规范化后的返回值
- [`DeriveCallDirections`](33-derive_call_directions.md) —— 后续基于本
  Pass 规范化的返回形态分析调用签名
- [PTO 代码生成总览](../codegen/00-pto_codegen.md) 与
  [编排代码生成](../codegen/01-orchestration_codegen.md) —— 直接消费规范
  化后的 `return[i] ↔ out_indices[i]` 映射
