# MaterializeTensorStrides Pass

将程序中所有 `view.has_value() && view.stride.empty()` 槽位按对应 layout 的 packed canonical 公式填入显式 stride（参考 RFC #1300 §2.4）。Pass 运行后即满足 codegen 入口契约：每个存在的 `TensorView` 都带显式 stride 与其 layout / shape 一致，严格模式 `TensorViewCanonical` verifier 也会通过。

> **状态**：本 Pass 已注册（`passes.materialize_tensor_strides()`）并通过独立单测，但**尚未接入默认 pipeline**。原因是按 canonical（逻辑 shape）公式物化 DN stride 与 PTO codegen 中遗留的「源 shape + 出口处再做 swap」路径冲突（`get_shape_source_idx` / `dn_swap`）。pipeline 接入将在 RFC #1300 P6/P7 的 codegen 清理一并完成；`pass_manager.py` 的预留位置已给后续阶段留下注释。

## 概述

PyPTO IR 上 `TensorType.tensor_view_` 当前可以处于两种等价形态：

- **隐式** —— `view.has_value() && view.stride.empty()`：layout 标签已设（如 `DN`），但每维 stride 为空。下游消费方需把空 stride 当作「该 layout 的 packed canonical stride」。
- **显式** —— 每个维度的 stride `ExprPtr` 都已写出。

为了让 codegen 看到单一可机械读取的契约，`MaterializeTensorStrides` 遍历整个程序，把所有隐式 `TensorView` 用 `tensor_view_semantics.h` 中的 `BuildLogicalStridesFromLayout` 改写为显式 packed canonical 形式。**裸 `TensorType`**（`!view.has_value()`）不被改写：`TensorViewCanonical` verifier 在弱/严格模式下都把它当作 ND-packed 接受，本身就无歧义。

**Requirements**：

- `SSAForm`、`SplitIncoreOrch`、`IncoreTileOps`、`TileOps2D`、`TileMemoryInferred`、`NormalizedStmtStructure`

**Produces**：

- `TensorViewCanonical` —— `PassPipeline` 在 Pass 之后自动用 registry 中的弱模式 verifier 校验

**默认 pipeline 中的位置**（自 RFC #1300 P6 起激活）：[`CanonicalizeIOOrder`](25-canonicalize_io_order.md) 与 [`InitMemRef`](27-init_memref.md) 之间。这是 codegen-prep 边界 —— 所有 layout-mutating pass（`LowerTransposeLoadParamLayout` / `ResolveBackendOpLayouts` / `ExpandMixedKernel` / `SplitVectorKernel`）已结束，`InitMemRef` 是第一个依赖显式 stride 的消费者。

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

2. **类型重写** —— `MaterializeType(type)`：
   - `TensorType` 满足 `view.has_value() && view.stride.empty() && layout != NZ`：用 `BuildLogicalStridesFromLayout(shape, layout)` 重建。其他 `TensorType` 形态原样返回（保持指针身份）。
   - `TensorType` 且 `layout == NZ`：原样返回（NZ 在 `TensorType` 上属于非法 IR；交由 verifier 报错而非在这里 `BuildLogicalStridesFromLayout` `CHECK`-fail）。
   - `TupleType`：递归处理元素类型；任一子类型变化时重建。
   - 其它：原样返回。

Pass **幂等** —— 在已物化的 IR 上重跑等于无操作（类型比较走指针身份就短路；无变化时 `MutableCopy` 也被跳过）。

| 行为 | 触发条件 |
| ---- | -------- |
| 用 packed canonical 填入 stride | `view.has_value() && view.stride.empty()` 且 `layout in {ND, DN}` |
| 原样直通 | `!view.has_value()`（裸 tensor） |
| 原样直通 | `view.has_value() && !view.stride.empty()`（已显式） |
| 原样直通 | `view.layout == NZ`（由 verifier 单独拒绝） |

## 示例

**Before** —— InCore 形参带有空 stride 的 DN view（用户写的 `pl.Tensor[..., pl.DN]` 未给显式 stride 提示）：

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
| `NZ` | 无法用 flat stride 表达（分形，仅 tile 使用）—— `BuildLogicalStridesFromLayout` `CHECK`-fail |

`MakeIndexMul` 对 `ConstInt * ConstInt` 做常量折叠（带 `__builtin_mul_overflow` 守卫，溢出时回退到符号 `Mul` 而不是静默 wrap），并消除 `× 1` 单位元；这样符号维保留为 `Mul` 表达式，静态常量链折叠为单个 `ConstInt`。

## 与 verifier 的协同

由于 Pass 声明 `produced = {... ∪ TensorViewCanonical}`，`PassPipeline` 在 Pass 完成后自动调用 registry 中的 `TensorViewCanonical` verifier；非法 IR（如 `TensorType` 上挂 NZ）会立即作为 `pypto::ValueError` 抛出。registry 默认是**弱模式** verifier（接受 `stride.empty()`）；**严格模式** verifier 通过 `passes.verify_tensor_view_canonical(program, require_materialized=True)` 显式调用，它就是 P6/P7 将启用的 codegen 入口契约。

## 相关

- [`CanonicalizeIOOrder`](25-canonicalize_io_order.md) —— 紧邻其前；产生本 Pass 消费的程序状态
- [`InitMemRef`](27-init_memref.md) —— 第一个依赖显式 stride 的下游消费者
- [`LowerTransposeLoadParamLayout`](18-lower_transpose_load_param_layout.md) —— 空 stride DN view 的主要来源（P6 起在默认 pipeline 中产生 canonical DN 参数）
- [`tensor_view_semantics.h`](../../../../include/pypto/ir/transforms/utils/tensor_view_semantics.h) —— 工具函数（`BuildLogicalStridesFromLayout` / `CheckCanonicalView` / `CanonicalizeView`）
- RFC [#1300](https://github.com/hw-native-sys/pypto/issues/1300) —— IR Tensor Layout 自洽表示方案
