# NormalizeReturnOrder Pass

将每个 InCore 函数的返回 tuple 重新排序，使 `return[i]` 对应声明顺序中的第
i 个 `Out`/`InOut` 参数，并相应地同步非 InCore 调用方中的结果
`TupleType`、绑定 `Var` 和 `TupleGetItemExpr` 索引。该 Pass 完成后，按位置消费
结果的下游逻辑会看到规范的 InCore tuple；编排（orchestration）代码生成则可直接
读取显式的「返回位置 → 参数」映射，无需追踪 `tile.store` / `ForStmt` yield 链。

## 概述

用户代码可以以任意顺序写 `tile.store` —— 先 `out_b` 后 `out_a`，或者与计算
混排。流水线前期会原样保留 body 顺序，因此 InCore 的 `ReturnStmt::value_`
中各输出可能与声明的 `Out`/`InOut` 参数顺序不一致。若不规范化，编排代码生
成就必须沿着每个 `return[i]` 反向追踪赋值与 `tile.store`，才能确定它实际写入
哪个参数 —— 这种分析应当属于 Pass，而不是代码生成层（参见
`docs/zh/dev/codegen/00-pto_codegen.md`）。

本 Pass 把契约规范化为「按位置 `return[k] ↔ out_indices[k]`」，分两步进行：

1. **Step A0（返回值参数化规范化）** —— 对每个 `InCore`、`Group`、
   `Spmd` 函数，把每个属于参数回写（param writeback）的 tensor 返回值改写为
   直接引用对应参数（指针同一性），追踪由共享的 `return_lineage` 工具完成。
   kernel 内部分配的输出（无法追踪到任何参数）和标量返回不受影响。
2. **Step A（InCore 函数重写）** —— 对每个 `InCore` 函数，计算一个使
   `ReturnStmt::value_` 与声明的 `Out`/`InOut` 参数顺序一致的置换，然后同步
   重写返回值与 `Function::return_types_`。
3. **Step B（调用端索引重映射）** —— 对每个非 InCore 函数（Orchestration /
   Group / Spmd / opaque），对每个调用 Step A 中已重排序函数的 `Call` /
   `Submit`，同步置换其结果 `TupleType`、创建匹配的绑定 `Var` 并按身份重映射
   所有使用，随后重写该结果上的所有 `TupleGetItemExpr.index_`。新索引为
   `permutation[old_index]`，因此观察者仍然把同名 SSA 变量绑定到同一物理输
   出。`Submit` 只置换被调函数返回值前缀；尾部 `Scalar[TASK_ID]` 保持原位。
   应用置换前，Pass 会验证候选被调函数在程序内的每个结果都只通过直接元素投影
   消费；whole-tuple 别名、控制流携带、返回值或调用参数会被拒绝，避免静默改变
   其契约。

恒等置换会跳过该被调函数的 Step A 与 Step B，但 Step A0 仍可能把可追踪的返回
别名替换成参数 Var。只有不含 `InCore`、`Group` 与 `Spmd` 函数的程序才是完整
no-op。

**流水线位置**: `Default` 策略中 #25 —— 位于 `StampTfreeSplit`（#24）之后，
`SkewCrossCorePipeline`（#26）、`LowerPipelineToSlots`（#27）与
`LowerPipelineLoops`（#28）之前。这样既保证所有 kernel 拆分 / tile 结构决策仍
基于原始返回顺序完成，又确保下游 tile 级 Pass（`CanonicalizeIOOrder`、
`InitMemRef`、`MemoryReuse`、`AllocateMemoryAddr`）以及最终的 PTO 编排代码生成
都能看到规范化后的顺序。

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
| 产出（Produced） | `ReturnParamsExplicit` |
| 失效（Invalidated） | — |

`SplitIncoreOrch` 保证 InCore 工作已经被外提为独立函数；`IncoreTileOps` 保
证函数体使用 tile 操作，从而 Step A 所依赖的
`tile.store(_, _, out_param)` 信号一定存在。本 Pass 产出
`ReturnParamsExplicit`（由 `verify_return_params_explicit.cpp` 校验）：每个
InCore/Group/Spmd 函数中属于参数回写的 tensor 返回值都以指针同一性引用对应
参数，编排代码生成因此只需查表即可建立返回值到实参的映射。本 Pass 不使任何
属性失效 —— SSA、规范化语句结构、内存推断等所有上游属性均被保留。

## 算法

### Step A0 —— 把返回值规范化为参数引用

对每个 `InCore` / `Group` / `Spmd` 函数，`CanonicalizeReturnValues` 调用
`return_lineage::ReturnedParamIndices`（可追踪 Var 到 Var 别名、循环携带、分支
值解析到同一参数的 tensor `IfStmt` 合流、builtin 回写、tuple 调用的
`TupleGetItem`，以及 Group/Spmd 包装函数调用），把每个可追踪到参数的 tensor
返回值替换为参数 `Var` 本身。无法追踪的值
（kernel 内部分配的输出）和标量保留原表达式。

**正是这一步让"返回 → 参数"映射无需分析即可读出。** 它运行之后，返回位置 `j`
写回参数 `i` 当且仅当 `ReturnStmt->value_[j]` 就**是** `params_[i]`（指针同一
性）——这正是 `IRProperty::ReturnParamsExplicit` 所断言的。因此位于本 pass 及其
之后的消费者（orchestration codegen、`ClassifyIterArgCarry`）调用
`return_lineage::ExplicitReturnedParamIndices(func)` 这一"函数局部的结构性读
取"，而不再重跑跨函数追踪器。`ReturnedParamIndices` 只保留给：在该属性建立**之
前**运行的调用方（`ExpandMixedKernel`、scope outliner）、本 pass 自身，以及必须
独立重新推导才能起到校验作用的属性验证器。

由于它是 codegen 的前置条件，手工构造 IR 并直接调用 orchestration codegen 的测试
必须先运行本 pass（见 `tests/ut/codegen/_orchestration_codegen_common.py`），正如
它们必须先运行 `DeriveCallDirections`、`MaterializeRuntimeScopes` 和
`ClassifyIterArgCarry` 一样。

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
四种情况之一时返回空置换（跳过 Step A，但保留 Step A0 已完成的规范化）：

- 函数体不含非空 `ReturnStmt`（开放 IR），或不含任何 Out/InOut 参数。
- `out_indices.size() > ret_to_param.size()` —— 声明的输出参数数量多于返回
  值数量，分析不完整，不能构造越界置换。
- 候选置换不是双射（目标重复、存在空洞或目标越界）。
- 计算出的置换是恒等置换（已规范）。

当置换非空时，`ReorderReturns` 通过 `MutableCopy` 克隆出新的 `Function`，
将末尾的 `ReturnStmt` 替换为
`value_[permutation[i]] = old_value_[i]` 的版本，并同步置换
`Function::return_types_`，使类型列表与值列表始终对齐。

### Step B —— 同步调用端 tuple 类型与投影

改写前，Pass 先只读扫描每个候选调用结果的绑定，并验证其所有使用都是非 InCore
调用方中的 `TupleGetItemExpr(binding, index)`。若候选调用是嵌套调用、位于 InCore
调用方，或其结果以 whole tuple 形式逃逸（经别名、`YieldStmt`/循环携带、
`ReturnStmt` 或调用参数），Pass 会在改动任何函数前报告带源码位置的错误。支持
这些形式需要显式构造逆置换 tuple 适配器；直接原地改变 tuple 顺序会在元素类型
相同时造成无提示的语义变化。

原子预检通过后，`TupleIndexPermutationMutator` 对每个非 InCore 函数执行一次
SSA 遍历：

- 对每个调用 Step A 中已重排序函数的 `Call(GlobalVar)` /
  `Submit(GlobalVar)`，按同一置换重建其结果 `TupleType`。对于 `Submit`，只移动
  前 `N = permutation.size()` 个元素；校验并保留末尾 `Scalar[TASK_ID]`，同时
  保留 `deps_`、attrs、kwargs、launch 字段与 predicate。
- 对有赋值目标的 call/submit 结果，创建带新 tuple 类型的绑定 `Var`，在 mutator
  的身份重映射中记录 `old_var → new_var`，并在 `reordered_tuple_vars_` 中记录
  `new_var → permutation_ref`。此后的所有使用都会指向同一个、类型正确的 tuple
  定义。
- 在记录同一 `Var` 的新定义前，先清理旧的身份重映射与追踪状态。RHS 会先被访问，
  因此 RHS 中的使用会看到前一个完整定义；随后新定义才替换旧的追踪状态。
- 对每个 `TupleGetItemExpr(tuple_var, k)`，若 `tuple_var` 在该映射中，把索
  引重写为 `permutation[k]`。

由于 Step A 重写函数签名与 Step B 重写调用端结果类型、绑定 Var 和索引在同一
次 Pass 调用中完成，出口程序保持类型自洽：每个 tuple 元素仍然绑定到同一个
物理输出 buffer，只是用新的下标访问。

## 约束

| 约束 | 原因 |
| ---- | ---- |
| Step A 仅重写 `InCore` 函数 | 其他函数类型（`Orchestration` / `Group` / `Spmd` / opaque）遵循用户声明的返回形态；它们的调用端在 Step B 中被重映射。`Group`/`Spmd` 的返回值在 Step A0 中仍会被规范化为参数引用，但不会被重排 |
| Step A0 不改动 kernel 内部分配的输出与标量 | 只有参数回写必须显式化；没有参数血缘的返回值没有可引用的参数 |
| `out_indices.size() > ret_to_param.size()` 时跳过 Step A | 不完整分析不能产生越界置换；保留 Step A0 已完成的参数引用规范化 |
| 恒等置换 ⇒ 不执行 Step-A 重排 | 避免不必要的重排克隆，同时保留 Step A0 已完成的参数引用规范化 |
| 被重排序调用的结果必须在非 InCore 调用方中仅通过直接 `TupleGetItemExpr` 投影消费 | Whole-tuple 别名、控制流携带、返回、参数及 InCore 调用点无法在本地安全重映射；Pass 会在任何函数被修改前拒绝这些形式 |
| Step B 追踪新类型绑定 `Var`，而不是改写前的定义 | 身份重映射会在投影查找前把所有使用指向新的 tuple 定义；每个新定义都会替换之前的追踪状态，避免失效绑定 |
| `Submit` 只置换被调函数返回值前缀 | 尾部 `Scalar[TASK_ID]` 属于任务启动语义，而不是被调函数返回值，必须保持最终 tuple 下标不变 |

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

同一组 SSA 赋值仍绑定到原来的物理输出：`a` 仍对应为 `out_b` 产生的值，`b`
仍对应为 `out_a` 产生的值；变化的只是 tuple 访问路径。`InOut` 参数的处理方式
相同。

代表性用例参见
`tests/ut/ir/transforms/test_normalize_return_order.py`：

- `test_swapped_returns_reordered` —— 上文展示的两个 Out 参数案例
- `test_already_ordered_noop` —— Step A 跳过恒等置换，Step A0 仍规范化参数回写
- `test_single_return_noop` —— 单个 Out 参数无需置换
- `test_non_incore_unchanged` —— 该纯 Orchestration 测试程序保持 no-op
- `test_three_returns_scrambled` —— 三元置换
- `test_2d_tensor_reorder` —— 2 维 tensor / 多维 offset
- `test_inout_param_reorder` —— `InOut` 参数同样参与重排

## 实现

**头文件**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass NormalizeReturnOrder();
```

**实现文件**: `src/ir/transforms/normalize_return_order_pass.cpp`

- `CanonicalizeReturnValues` —— Step A0 改写器：通过
  `return_lineage::ReturnedParamIndices` 把可追踪的 tensor 返回值替换为参
  数 `Var`。
- `BuildReturnToParamMapping` —— Step A 分析：遍历函数体，将每个
  `ReturnStmt` 值反向映射到 Out/InOut 参数下标。
- `CollectOutIndices` —— 收集 `ParamDirection` 为 `Out` / `InOut` 的参数
  位置。
- `ComputeReturnPermutation` —— 综合上述两个分析，得到最终的
  `permutation[old_index] = new_index`；不需重写或分析不完整时返回空。
- `ReorderReturns` —— 基于 `MutableCopy(func)` 构造新的 `Function`，置换
  `ReturnStmt::value_` 与 `Function::return_types_`。
- `FindUnsafeReturnPermutations` —— 在 Step A 前预检每个候选调用结果的使用，
  并报告不受支持的 whole-tuple 或 InCore 调用方形式。
- `TupleIndexPermutationMutator` —— Step B 改写器：置换 `Call` / `Submit`
  结果 tuple 类型，按身份重映射其绑定 Var，并在保留 `Submit` TASK_ID 尾部的
  同时重写 `TupleGetItemExpr` 索引。

**属性**: `include/pypto/ir/transforms/pass_properties.h`

```cpp
inline const PassProperties kNormalizeReturnOrderProperties{
    .required = {IRProperty::SplitIncoreOrch, IRProperty::IncoreTileOps},
    .produced = {IRProperty::ReturnParamsExplicit}};
```

**Python 绑定**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("normalize_return_order", &pass::NormalizeReturnOrder,
           "Create a return order normalization pass\n\n"
           "Canonicalizes tensor param-writeback returns and reorders InCore return tuples\n"
           "to Out/InOut parameter order. Reordered Call/Submit results must be directly\n"
           "bound and used only through TupleGetItem projections in non-InCore callers.");
```

**类型存根**: `python/pypto/pypto_core/passes.pyi`

```python
def normalize_return_order() -> Pass:
    """Create a return-order normalization pass.

    Reordered Call/Submit results must be directly bound and used only through
    tuple-element projections in non-InCore callers.
    """
```

**测试**: `tests/ut/ir/transforms/test_normalize_return_order.py`

## 相关

- [`OutlineInCoreScopes`](08-outline_incore_scopes.md) —— 上游产出本 Pass
  改写的 `InCore` 函数
- [`SkewCrossCorePipeline`](27-skew_cross_core_pipeline.md) 与
  [`LowerPipelineToSlots`](28-lower_pipeline_to_slots.md) —— 在两者之间运行，
  各自认领自己处理的流水线循环
- [`LowerPipelineLoops`](29-lower_pipeline_loops.md) —— 展开上述两个 pass 未接手的
  流水线作用域时，消费规范化后的返回值
- [`DeriveCallDirections`](38-derive_call_directions.md) —— 后续基于本
  Pass 规范化的返回形态分析调用签名
- [PTO 代码生成总览](../codegen/00-pto_codegen.md) 与
  [编排代码生成](../codegen/01-orchestration_codegen.md) —— 消费显式的
  「返回位置 → 参数」映射
