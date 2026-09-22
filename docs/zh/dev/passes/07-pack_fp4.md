# PackFp4 Pass

将前端逻辑 FP4 nibble 类型打包为 PTOAS 可用的 `FP4E2M1X2`。

**限制、cast 策略与待办**见 [FP4](../fp4.md)。本页只写 pass 算法。

## 概述

前端 IR 把 `pl.FP4` 计为逻辑 4-bit nibble。PTOAS 只认识 `!pto.f4E2M1x2`
packed pair。本 pass 改写 dtype、末维 covering size / offset、ND leading
stride 及相关 call 操作数，使后续 pass 与 tile_buf codegen 使用 packed
carrier。在 PTOAS 边界，单卡 codegen 再把 GM view / partition 末维几何扩回
nibble，供 pto-isa `GetByteSize`。

**范围（仅静态）**：末维 covering size、offset、leading stride 必须是
`ConstInt`（size/stride 为正偶数；offset 为偶数）。动态末维几何**硬拒**。
优先 UINT8 半宽 cache ABI 或手写 `pl.FP4E2M1X2` — 见 [FP4](../fp4.md)。

**依赖**：`SSAForm`、`NoNestedCalls`、`NormalizedStmtStructure`。

**何时使用**：Default 流水线中紧跟 `FlattenCallExpr`，在 Outline /
`ConvertTensorToTileOps` / `MaterializeTensorStrides` 之前；只跑一次。

## 本 pass 之后的单位约定

Carrier 与 nibble 的完整表见 [FP4 单位约定](../fp4.md#单位约定)。

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::PackFp4()` | `passes.pack_fp4()` | Function-level |

```python
from pypto.pypto_core import passes

packed = passes.pack_fp4()(program)
```

## 流水线位置

```text
... -> Simplify -> FlattenCallExpr -> PackFp4
    -> OutlineHierarchyScopes -> ... -> ConvertTensorToTileOps -> ...
    -> LegalizeTileCast -> ...
```

## 算法

1. **类型改写**：把仍携带逻辑 FP4 的 `Tensor` / `Tile` / `Buffer` /
   `MultiBuffer` 打成 `FP4E2M1X2`。`DistributedTensor` FP4 与标量 FP4
   **硬拒**（见 [FP4](../fp4.md)）。
2. **末维 covering size**（`shape`、`valid_shape`、slice / load size）：正偶数
   `ConstInt` → `K / 2`。
3. **末维 offset**：偶数 `ConstInt` → `offset / 2`。
4. **ND leading stride**：要求末轴连续（`stride[-1] == 1`）；leading stride 按
   covering-size 规则打包。
5. **Call 操作数**：白名单改写（slice、load、store、assemble、**view** 含可选
   `valid_shape`、create、full 等）。**拒绝** reshape / transpose / read /
   write / `tensor.dim` / remote / window / put / get。未列入白名单、且参数为
   非空坐标 tuple **或** 整型 / `INDEX` 标量坐标的 op 大声失败。
6. **布局守卫**：拒绝 DN / NZ、col-major tile、cube 内存。

## 示例

**Before**（逻辑 FP4，末维 64 nibble）：

```python
x: pl.Tensor[[16, 64], pl.FP4]
tile = pl.load(x, [0, 2], [16, 32])
```

**After PackFp4**：

```python
x: pl.Tensor[[16, 32], pl.FP4E2M1X2]
tile = pl.load(x, [0, 1], [16, 16])
```
