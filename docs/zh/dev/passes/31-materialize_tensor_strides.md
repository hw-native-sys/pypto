# MaterializeTensorStrides Pass

将程序中所有 `TensorType` / `DistributedTensorType` 上的 `view.has_value() && view.stride.empty()` 槽位按对应 layout 的 packed canonical 公式填入显式 stride（参考 RFC #1300 §2.4）。Pass 运行后即满足 codegen 入口契约：每个存在的 `TensorView` 都带显式 stride 与其 layout / shape 一致，严格模式 `TensorViewCanonical` verifier 也会通过。

> **状态**：本 Pass 已注册（`passes.materialize_tensor_strides()`）、有单测覆盖，并自 RFC #1300 P6 起接入默认 tile/PTO pipeline，位置在 `CanonicalizeIOOrder` 与 `InitMemRef` 之间。

## 概述

PyPTO IR 上 `TensorType.tensor_view_` 当前可以处于两种等价形态：

- **隐式** —— `view.has_value() && view.stride.empty()`：layout 标签已设（如 `DN`），但每维 stride 为空。下游消费方需把空 stride 当作「该 layout 的 packed canonical stride」。
- **显式** —— 每个维度的 stride `ExprPtr` 都已写出。

为了让 codegen 看到单一可机械读取的契约，`MaterializeTensorStrides` 遍历整个程序，把所有隐式 `TensorView` 用 `tensor_view_semantics.h` 中的 `BuildLogicalStridesFromLayout` 改写为显式 packed canonical 形式。**裸 `TensorType`**（`!view.has_value()`）不被改写：`TensorViewCanonical` verifier 在弱/严格模式下都把它当作 ND-packed 接受，本身就无歧义。输入类型是 `DistributedTensorType` 时，重建后的类型仍保持 distributed wrapper，并保留 `memref`、`TensorView.pad` 等非 stride 元数据以及 `window_buffer` 反向引用。

**Requirements**：

- `SSAForm`、`SplitIncoreOrch`、`IncoreTileOps`、`TileOps2D`、`TileMemoryInferred`、`NormalizedStmtStructure`

**Produces**：

- `TensorViewCanonical` —— `PassPipeline` 在 Pass 之后自动用 registry 中的**严格模式** verifier 校验（拒绝 `view.has_value() && stride.empty()` —— 正是本 Pass 负责消除的状态）

**默认 pipeline 中的位置**（自 RFC #1300 P6 起激活）：[`CanonicalizeIOOrder`](30-canonicalize_io_order.md) 与 [`InitMemRef`](32-init_memref.md) 之间。这是 codegen-prep 边界 —— 所有 layout-mutating pass（`ResolveBackendOpLayouts` / `ExpandMixedKernel` / `SplitVectorKernel`）已结束，`InitMemRef` 是第一个依赖显式 stride 的消费者。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::MaterializeTensorStrides()` | `passes.materialize_tensor_strides()` | Program-level |

```python
from pypto.pypto_core import passes

mat_pass = passes.materialize_tensor_strides()
program_canon = mat_pass(program)
```

## 算法

Pass 使用带 Var 替换缓存的 `IRMutator`，结构与 `InferTileMemorySpace` 一致。它遍历程序中可达的每个 `TypePtr`：

1. **逐函数重建** —— 重新构造形参 / 返回类型 / 函数体：
   - 遍历形参类型；若某形参 `TensorType` 经 `MaterializeType` 后变为不同的类型，构造新 `Var`（保留 `name_hint` 与 `span`）并登记替换。
   - 同理处理返回类型。
   - 通过 `IRMutator::VisitStmt` 遍历函数体：
     - `VisitExpr_(VarPtr)`：若该 Var 的类型经 `MaterializeType` 改变，构造新 Var（查 `var_cache_`，确保对同一个原 Var 的所有引用都解析到同一新 Var）。
     - `VisitExpr_(IterArgPtr)`：与 Var 同理，附加递归处理 `init_value_`。
     - `VisitExpr_(CallPtr)`：注册 op 走 `OpRegistry` 重建；`GlobalVar` 调用 / 未注册 op 走直接 `Call` 构造路径。
     - `VisitStmt_(AssignStmtPtr)`：先重建 RHS；若 RHS Call 的返回类型比 LHS Var 当前类型更显式（已物化），同步 LHS Var。

2. **类型重写** —— `MaterializeType(type, span)`：
   - `TensorType` / `DistributedTensorType` 且 `layout == NZ` 但 shape **未分块**：无论 stride 是否显式，一律用 `INTERNAL_CHECK_SPAN` **拒绝**。NZ 在 tensor 类型上是合法的，但只允许 [BlockNzTensorViews](14-block_nz_tensor_views.md) 产出的分块 rank-(r+2) 形式 `[..., C/c0, R/16, 16, c0]` —— 只有这个 shape 下，下面构建的行主序 stride 才真正描述 NZ 字节序。未分块就到达这里意味着 pass 14 没有运行或漏掉了槽位，因此这是 pass 顺序不变量而非用户错误（面向用户的对齐诊断在 `BlockNzShape` 中）。`span` 参数（携带该类型的 `Var` / `IterArg` / `Call` / `Submit` / 形参 / 函数节点）用于在报错信息中定位出问题的标注。
   - `TensorType` / `DistributedTensorType` 满足 `view.has_value() && view.stride.empty()`：用 `BuildLogicalStridesFromLayout(shape, layout)` 重建，并保留 distributed wrapper 与可选元数据（`memref`、`TensorView.pad`、`window_buffer`）。其他 tensor 形态原样返回（保持指针身份）。
   - `TupleType`：递归处理元素类型（沿用同一个 `span`）；任一子类型变化时重建。
   - 其它：原样返回。

NZ 的拒绝逻辑放在 pass 内部，而不是交给配对的 verifier。交给 verifier 会让这条拒绝取决于验证是否开启：在 `PYPTO_VERIFY_LEVEL=none` 下，pass 会原样返回这个非法槽位，却仍然声明产出 `TensorViewCanonical`，非法 layout 只会在下游以晦涩的后端 layout mismatch 形式重新冒出来。

Pass **幂等** —— 在已物化的 IR 上重跑等于无操作（类型比较走指针身份就短路；无变化时 `MutableCopy` 也被跳过）。

| 行为 | 触发条件 |
| ---- | -------- |
| 用 packed canonical 填入 stride | `view.has_value() && view.stride.empty()` 且 `layout in {ND, DN}` |
| 原样直通 | `!view.has_value()`（裸 tensor） |
| 原样直通 | `view.has_value() && !view.stride.empty()` 且 `layout in {ND, DN}`（已显式） |
| 拒绝（`InternalError`） | `view.layout == NZ` 且 shape 未分块，无论 stride 是否显式（BlockNzTensorViews 本应已分块） |

各行互斥：NZ 检查先执行，因此显式 stride 的 NZ view 会被拒绝，而不是原样直通。

## 示例

**Before** —— InCore 形参带有空 stride 的 DN view（写 `pl.TensorView(layout=DN)` 但未给显式 stride 提示）：

```python
@pl.function(type=pl.FunctionType.InCore)
def kernel(b: pl.Tensor[[2, 4, 8], pl.FP32, pl.TensorView(stride=[], layout=pl.TensorLayout.DN)],
           out: pl.Out[pl.Tensor[[2, 4, 8], pl.FP32]]) -> pl.Tensor[[2, 4, 8], pl.FP32]:
    ...
```

**After**：

```python
@pl.function(type=pl.FunctionType.InCore)
def kernel(b: pl.Tensor[[2, 4, 8], pl.FP32, pl.TensorView(stride=[32, 1, 4], layout=pl.TensorLayout.DN)],
           out: pl.Out[pl.Tensor[[2, 4, 8], pl.FP32]]) -> pl.Tensor[[2, 4, 8], pl.FP32]:
    ...
```

shape `[2, 4, 8]` 的 DN packed canonical stride：

- `stride[1] = 1`（DN 内层对中较小的那一维）
- `stride[2] = shape[1] = 4`
- `stride[0] = shape[1] * shape[2] = 32`

ND 情况下公式退化为标准行主序 packed stride。

## Stride 公式

详见 [`tensor_view_semantics.h`](../../../../include/pypto/ir/transforms/utils/tensor_view_semantics.h) 中的 `BuildLogicalStridesFromLayout`。

| Layout | 公式 |
| ------ | ---- |
| `ND` | `stride[n-1] = 1; stride[k] = stride[k+1] * shape[k+1]`，`k = n-2 .. 0` |
| `DN`（`n ≥ 2`） | `stride[n-2] = 1`；`stride[n-1] = shape[n-2]`；`stride[n-3] = shape[n-2] * shape[n-1]`；`stride[k] = stride[k+1] * shape[k+1]`，`k = n-4 .. 0` |
| `NZ` | 对*分块* shape 求行主序 —— 与 ND 规则相同。对 `[..., C/c0, R/16, 16, c0]` 求行主序精确复现 pto-isa 的 `BaseShape2D<..., Layout::NZ>`，因此 NZ 不需要自己的规则。未分块的 NZ shape 在此之前已被拒绝。 |

`MakeIndexMul` 对 `ConstInt * ConstInt` 做常量折叠（带 `__builtin_mul_overflow` 守卫，溢出时回退到符号 `Mul` 而不是静默 wrap），并消除 `× 1` 单位元；这样符号维保留为 `Mul` 表达式，静态常量链折叠为单个 `ConstInt`。

## 与 verifier 的协同

由于 Pass 声明 `produced = {... ∪ TensorViewCanonical}`，`PassPipeline` 在 Pass 完成后自动调用 registry 中的 `TensorViewCanonical` verifier。registry 默认是**严格模式** verifier（RFC #1300 §2.4 codegen 入口契约）：它拒绝 `view.has_value() && stride.empty()` —— 因为本 Pass 就是负责物化这些 stride 的。裸 `TensorType`（`!view.has_value()`）仍然接受 —— 隐式 ND-packed 自然 canonical。同一 verifier 也可通过 `passes.verify_tensor_view_canonical(program, require_materialized=True)` 显式调用；传 `require_materialized=False` 切换到弱模式（用于物化之前的解析期 / 前期 pass 窗口）。

verifier 是配对复核，而不是唯一的强制点。验证本身可配置（`PYPTO_VERIFY_LEVEL` / `PassContext`），随时可能关闭，因此 Pass 声明产出的每条不变量都由 Pass 自己建立 —— 上面的 NZ 拒绝也在其中。verifier 随后再复核一遍，用于捕捉本 Pass 以及下游任何改写 tensor 类型的代码引入的回归。

## 相关

- [`CanonicalizeIOOrder`](30-canonicalize_io_order.md) —— 紧邻其前；产生本 Pass 消费的程序状态
- [`InitMemRef`](32-init_memref.md) —— 第一个依赖显式 stride 的下游消费者
- [`tensor_view_semantics.h`](../../../../include/pypto/ir/transforms/utils/tensor_view_semantics.h) —— 工具函数（`BuildLogicalStridesFromLayout` / `CheckCanonicalView` / `CanonicalizeView`）
- RFC [#1300](https://github.com/hw-native-sys/pypto/issues/1300) —— IR Tensor Layout 自洽表示方案
