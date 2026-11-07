# NormalizeReturnOrder Pass

`Orchestration` 中属于参数回写的张量返回在 `InitMemRef` 前固定为显式参数引用，供 HOST 代码生成直接读取；返回顺序保持不变。内存分配后 SSA 别名的分配元数据可能不同，不能再依赖后期追溯。原有形状、视图与类型检查保持不变。

完整的 `Submit` 结果是 `(results..., TaskId)`，不是张量别名。只有显式的张量
投影才能被参数化；返回整个容器时保留完成句柄和元组类型。循环回写证明会在循环
携带假设下，为当前查询中已确定的别名前缀设置暂定来源，因此从展开后的 store
开始查询不会把合法的归纳证明误判为缓存失败。这不表示新分配或重新绑定的循环
结果可以成为初始 buffer 的别名。

缓存的循环证明只保留初值或 yield 实际读取的外层假设，独立证明不会随无关的
外层失败而失效。可合并的左偏堆保留所有依赖：完成证明时选择最深的真实依赖，
再将剩余要求合并到该外层证明；每条依赖边的额外处理为对数复杂度。词法嵌套
本身不构成依赖。

存储根查询还可以将尾部的局部定义指定为根，并显式给出前部真实参数的数量。
真实参数仍接受重绑定检查；局部 tile 定义保留存储身份，使别名返回共享同一个
输出 buffer 和 store。

严格缓存覆盖一次 `DistributedCodegen::Generate` 调用，因此不同 CHIP 入口可复用
共享 helper 的映射。正常返回或异常退出都会清空缓存，不在调用之间保留分析状态。

原始 HOST 返回值绑定使用独立的严格追踪缓存策略：返回来源分析实际访问的每个
Host helper 必须没有 return（保留隐式输出路径），或只有一个末尾 return。末尾的
AUTO/MANUAL runtime scope 都是无条件容器；条件返回或多条 return 必须先完成
控制流规范化，再进行 HOST 代码生成。普通 Call 可以逐项绑定 tuple 返回值。
显式 CHIP 张量返回无法确定存储身份时，不能回退到按可写参数顺序绑定。

设备 kernel 与 Group/SPMD wrapper 保留已有的回写返回契约：混合核拆分后，
return 可以分布在互斥的核分支中。统一分析这些返回，只有每个位置的存储身份
一致时才固定为参数；Host/helper 的返回控制流检查仍然保留。

在控制流规范化前，非严格的来源分析也会将条件返回或多条返回的 helper 来源
保留为未知，避免调用方的返回参数化掩盖它们，绕过后续严格检查。

将每个 InCore 函数的返回 tuple 重新排序，使 `return[i]` 对应声明顺序中的第
i 个 `Out`/`InOut` 参数，并相应地重映射非 InCore 调用方中的
`TupleGetItemExpr` 索引。该 Pass 完成后，编排（orchestration）代码生成可以通
过位置直接映射 `out_indices[i]`，无需追踪 `tile.store` / `ForStmt` yield
链。

外部 AIC/AIV 声明没有 `ReturnStmt`，其 `external_return_params` 中按顺序排列的参数
引用提供相同的返回身份契约；来源追溯与结构化读取都会验证该契约。没有此属性时，
声明必须按参数顺序返回全部可写参数，且返回数量和 Tensor 类型匹配。有歧义的契约
会被拒绝。经过 extern 调用的 wrapper 返回值同样由步骤 A0 规范化。
参见[外部 kernel](../language/04-external-kernels.md)。

## 概述

用户代码可以以任意顺序写 `tile.store` —— 先 `out_b` 后 `out_a`，或者与计算
混排。流水线前期会原样保留 body 顺序，因此 InCore 的 `ReturnStmt::value_`
中各输出可能与声明的 `Out`/`InOut` 参数顺序不一致。若不规范化，编排代码生
成就必须沿着每个 `return[i]` 反向追踪赋值与 `tile.store`，才能确定它实际写入
哪个参数 —— 这种分析应当属于 Pass，而不是代码生成层（参见
`docs/zh/dev/codegen/00-pto_codegen.md`）。

本 Pass 把契约规范化为「按位置 `return[k] ↔ out_indices[k]`」，分两步进行：

1. **Step A0（返回值参数化规范化）** —— 对每个 `InCore`、`Group`、
   `Spmd`、`Graph`、`Orchestration` 函数，把每个属于参数回写（param writeback）的 tensor 返回值改写为
   直接引用对应参数（指针同一性），追踪由共享的 `return_lineage` 工具完成。
   kernel 内部分配的输出（无法追踪到任何参数）和标量返回不受影响。
2. **Step A（InCore 函数重写）** —— 对每个 `InCore` 函数，计算一个使
   `ReturnStmt::value_` 与声明的 `Out`/`InOut` 参数顺序一致的置换，然后同步
   重写返回值与 `Function::return_types_`。
3. **Step B（调用端索引重映射）** —— 对每个非 InCore 函数（Orchestration /
   Group / Spmd / Graph / opaque），重写所有 `TupleGetItemExpr.index_`，前提是它的
   tuple 操作数来源于 Step A 中被重排序的函数调用结果。新索引为
   `permutation[old_index]`，因此调用结果上的观察者仍然把同名 SSA 变量绑
   定到同一物理输出。

对于 tensor 返回值已经直接指名形参、且返回顺序已经与 `Out`/`InOut` 参数声
明顺序匹配的函数，本 Pass 是 **no-op**；对于不含 `InCore`、`Group`、`Spmd`、
`Graph`、`Orchestration` 函数的程序也是 no-op —— Step A0 覆盖这五类，所以「不含 InCore 函数」
本身已不足以说明本 Pass 什么都不做。

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
| 产出（Produced） | `ReturnParamsExplicit` |
| 失效（Invalidated） | — |

`SplitIncoreOrch` 保证 InCore 工作已经被外提为独立函数；`IncoreTileOps` 保
证函数体使用 tile 操作，从而 Step A 所依赖的
`tile.store(_, _, out_param)` 信号一定存在。本 Pass 产出
`ReturnParamsExplicit`（由 `verify_return_params_explicit.cpp` 校验）：每个
InCore/Group/Spmd/Graph/Orchestration 函数中属于参数回写的 tensor 返回值都以指针同一性引用对应
参数，编排代码生成因此只需查表即可建立返回值到实参的映射。本 Pass 不使任何
属性失效 —— SSA、规范化语句结构、内存推断等所有上游属性均被保留。

## 算法

### Step A0 —— 把返回值规范化为参数引用

对每个 `InCore` / `Group` / `Spmd` / `Graph` / `Orchestration` 函数，本 pass 使用
`return_lineage::ReturnedParamIndices`（可追踪 Var 到 Var 别名、循环携带、
builtin 回写、tuple 调用的 `TupleGetItem`，以及 Group/Spmd 包装函数调用），
把每个可追踪到参数的 tensor 返回值替换为参数 `Var` 本身。无法追踪的值
（kernel 内部分配的输出）和标量保留原表达式。

张量 `IfStmt` 合流仅在所有分支都追踪到同一个参数时才能继续追踪。存储来源一致性
与合流元数据分别保留，因此两个分支内部的收窄可以在合流后恢复。合流元数据取自
追踪后的分支类型，包括已验证调用转发的 caller 类型。实际返回的张量
必须与参数类型一致，包括视图（view）和有效形状
（valid shape）元数据；合流后的 `tensor.set_validshape` 结果不能通过规范化丢失这些
元数据。来源冲突、新分配张量或改变视图的分支保持原返回表达式。Graph 函数继续
拒绝通过合流追踪返回值：其边界分析尚不支持合流张量的来源，接受这种返回值可能
导致 replay 时冻结随调用变化的边界切片。变量的参数根和
失败结果在分支及多个返回值之间共享缓存，避免连续条件回写重复追踪前驱合流。
回写链检查最终结果的类型：临时收窄有效形状后用 `tensor.assemble` 恢复，仍可将
返回值规范化为参数引用。

**正是这一步让"返回 → 参数"映射无需分析即可读出。** 它运行之后，返回位置 `j`
写回参数 `i` 当且仅当 `ReturnStmt->value_[j]` 就**是** `params_[i]`（指针同一
性）——这正是 `IRProperty::ReturnParamsExplicit` 所断言的。因此位于本 pass 及其
之后的消费者（orchestration codegen、`ClassifyIterArgCarry`）调用
`return_lineage::ExplicitReturnedParamIndices(func)` 这一"函数局部的结构性读
取"，而不再重跑跨函数追踪器。`ReturnedParamIndices` 只保留给：在该属性建立**之
前**运行的调用方（`ExpandMixedKernel`、scope outliner）、本 pass 自身，以及必须
独立重新推导才能起到校验作用的属性验证器。

`Graph` 被纳入该集合的理由与包装函数相同：编排代码生成正是通过这张映射把 Graph
调用的结果别名到调用端张量。Graph 的函数体写作 `out = core(..., out, ...); return
out`，outliner 会把它变成 `TupleGetItem` 重绑定，因此返回值到达 codegen 时是一个
SSA 重命名。在 Graph 被排除期间，这张映射全部返回 `nullopt`，codegen 退回到位置
启发式（返回位置 `j` ↦ 第 `j` 个 `Out`/`InOut` 参数）；一旦某个 Graph 的返回顺序
与参数顺序不一致，它的每个结果都会被静默绑定到错误的张量，且没有任何诊断
（#2601）。

由于它是 codegen 的前置条件，手工构造 IR 并直接调用 orchestration codegen 的测试
必须先运行本 pass（见 `tests/ut/codegen/_orchestration_codegen_common.py`），正如
它们必须先运行 `DeriveCallDirections`、`MaterializeRuntimeScopes` 和
`ClassifyIterArgCarry` 一样。

### Step A —— 计算并应用每个函数的置换

Step A 复用 Step A0 的返回值到参数映射。设备 kernel 存在多个返回点时，只有
所有路径的同一返回位置都指向同一参数，映射才记录该参数。来源未知的位置保留原索引。

`ComputeReturnPermutation` 将每个已知位置映射到对应可写参数在
`CollectOutIndices(func)` 中的索引。返回数量不匹配、置换不满足双射或本身是恒等
置换时跳过。重排时，`ReorderReturns` 对每个返回点和 `Function::return_types_`
应用同一置换。属性验证器同样检查每个返回点。

### Step B —— 重映射调用端的 `TupleGetItemExpr`

对（Step A 已重写后的）程序中的每个非 InCore 函数，
`TupleIndexPermutationMutator` 在单次 SSA 遍历中：

- 对每个 RHS 是 `Call(GlobalVar)` 或 `Submit(GlobalVar)`、且被调函数已被重排的
  `AssignStmt`，同步置换表达式的元组元素类型，并创建使用该类型的新结果 `Var`。
  置换的是调用端已实例化的类型，因此保留 caller 的局部 shape 符号。
  Submit 末尾的 `TASK_ID`、依赖及启动元数据保持不变。
- 在 `reordered_tuple_vars_` 中记录旧结果 `Var`、新绑定和置换。
- 对 SSA 转换前具有多个定义的绑定，将每次重排调用写入新的物理临时变量，再将逻辑元组
  写回原绑定。分支和循环因此保留可供 `ConvertToSSA` 合并的共享身份，也支持重排调用与
  非重排赋值混用。
- 一旦该 `Var` 被重新赋值（赋给非重排序函数的调用、非 Call 表达式等），立即
  移除其条目，避免基于身份的查找读到失效绑定。
- 对每个 `TupleGetItemExpr(tuple_var, k)`，若 `tuple_var` 在该映射中，把索
  引重写为 `permutation[k]`，并使用新的元组绑定。
- 整元组使用（包括别名和转发）通过投影构造原逻辑顺序的元组；每个定义缓存一次，
  不重复执行调用。声明多个平铺返回位置的函数会把逻辑元组展开为对应返回值。

由于 Step A 重写函数签名与 Step B 重写调用端索引在同一次 Pass 调用中完成，
出口程序状态自洽：每个 tuple 元素仍然绑定到同一个物理输出 buffer，只是用
新的下标访问；即使输出的 dtype 或 shape 不同，调用、绑定和投影类型也保持一致。

## 约束

| 约束 | 原因 |
| ---- | ---- |
| Step A 仅重写 `InCore` 函数 | 其他函数类型（`Orchestration` / `Group` / `Spmd` / `Graph` / opaque）遵循用户声明的返回形态；它们的调用端在 Step B 中被重映射。`Group`/`Spmd`/`Graph`/`Orchestration` 的返回值在 Step A0 中仍会被规范化为参数引用，但不会被重排 |
| Step A0 在 Graph 函数之外要求类型兼容的分支一致性（branch consensus） | 所有分支必须追踪到同一参数；最终返回张量必须保留参数的类型、视图和有效形状。允许恢复中间的收窄。Graph 合流返回值仍不支持 |
| Step A0 不改动 kernel 内部分配的输出与标量 | 只有参数回写必须显式化；没有参数血缘的返回值没有可引用的参数 |
| `out_indices.size() > ret_to_param.size()` 时跳过 | 不完整分析不能产生越界置换 —— 保留原状，让 verifier 捕获不一致 |
| 恒等置换 ⇒ 不重写 | 避免不必要的 `Function` 克隆，使 Pass 幂等 |
| Step B 记录原结果 `Var` 并创建带有新类型的绑定 | 投影使用新的物理元组，其他消费者保留旧逻辑顺序；不修改现有 IR 节点 |

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
        return (out_a, out_b)        # ReorderReturns: 置换 [1, 0]

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, out_a, out_b):
        ret = self.kernel(x, out_a, out_b)
        a = ret[1]                                # TupleIndexPermutationMutator: 0 → 1
        b = ret[0]                                # TupleIndexPermutationMutator: 1 → 0
        return (out_b, out_a)
```

SSA 变量 `a` 仍引用 `out_b`，`b` 仍引用 `out_a`；变化的只是 tuple 投影索引。
编排函数直接返回这两个相同缓冲区，保留原有的 `(out_b, out_a)` 结果顺序。
`InOut` 参数的处理方式相同。

完整用例参见
`tests/ut/ir/transforms/test_normalize_return_order.py`：

- `test_swapped_returns_reordered` —— 上文展示的两个 Out 参数案例
- `test_already_ordered_noop` —— 已规范的 IR 保持不变
- `test_single_return_noop` —— 单个 Out 参数无需置换
- `test_non_incore_unchanged` —— opaque 张量计算函数保持不变
- `test_three_returns_scrambled` —— 三元置换
- `test_2d_tensor_reorder` —— 2 维 tensor / 多维 offset
- `test_inout_param_reorder` —— `InOut` 参数同样参与重排
- `test_branch_writeback_and_wrapper_return_reference_param` —— 同一缓冲区的分支回写及 wrapper 转发（#2966）
- `test_dynamic_branch_call_returns_reference_caller_param` —— 生成的 phi 类型保留 callee 局部形状变量
- `test_branch_return_with_unproven_identity_is_not_canonicalized` —— 来源冲突、新张量、视图及有效形状分支
- `test_writeback_after_branch_merge_preserves_returned_valid_shape` —— 两种返回顺序下保留合流后收窄的元数据
- `test_restored_writeback_metadata_references_param` —— 两种返回顺序下，共享血缘的恢复结果与收窄结果
- `test_sequential_branch_writebacks_share_lineage` —— 48 次连续合流共享血缘缓存

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
- `TupleIndexPermutationMutator` —— Step B 改写器：跟踪调用结果变量，并
  重写 `TupleGetItemExpr` 索引。

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

- [`OutlineInCoreScopes`](09-outline_incore_scopes.md) —— 上游产出本 Pass
  改写的 `InCore` 函数
- [`SkewCrossCorePipeline`](30-skew_cross_core_pipeline.md) 与
  [`LowerPipelineToSlots`](31-lower_pipeline_to_slots.md) —— 在两者之间运行，
  各自认领自己处理的流水线循环
- [`LowerPipelineLoops`](32-lower_pipeline_loops.md) —— 展开上述两个 pass 未接手的
  流水线作用域时，消费规范化后的返回值
- [`DeriveCallDirections`](45-derive_call_directions.md) —— 后续基于本
  Pass 规范化的返回形态分析调用签名
- [PTO 代码生成总览](../codegen/00-pto_codegen.md) 与
  [编排代码生成](../codegen/01-orchestration_codegen.md) —— 直接消费规范
  化后的 `return[i] ↔ out_indices[i]` 映射
