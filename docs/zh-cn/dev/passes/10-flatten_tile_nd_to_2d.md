# FlattenTileNdTo2D Pass

将 InCore 函数中的 ND Tile 操作（3D+）展平为 2D，合并除最后一个维度外的所有维度。

## 概述

PTO-ISA 仅支持 2D Tile。`ConvertTensorToTileOps` 之后，Tile 可能是 ND（匹配张量形状）。该 Pass 通过将高维轴合并为一个维度并保持最后一个轴不变，将所有 >2D 的 Tile 操作展平为 2D。例如，Tile `[2, 3, 4]` 变为 `[6, 4]`。

**前置条件**：

- 输入 IR 必须为 SSA 形式
- 输入 IR 必须包含 Tile 操作（需先运行 `ConvertTensorToTileOps`）
- 所有 Tile 维度必须为静态（`ConstInt`）
- 所有 Tile 归约操作必须沿最后一个轴归约
- 所有 Tile 内存必须是连续的

**使用时机**：在 `ConvertTensorToTileOps` 之后、`ExpandMixedKernel` / `InitMemRef` 之前运行。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::FlattenTileNdTo2D()` | `passes.flatten_tile_nd_to_2d()` | 函数级 |

**Python 用法**：

```python
from pypto.pypto_core import passes

flatten_pass = passes.flatten_tile_nd_to_2d()
program_2d = flatten_pass(program)
```

## 算法

对每个 InCore 函数（InCore、AIC、AIV）：

1. **验证前置条件**：检查静态形状、最后轴归约、不允许对 >2D 使用 `tile.read`/`tile.write`/`tile.slice`
2. **变换语句**：遍历函数体，将 >2D Tile 操作转换为 2D

按语句类型处理：

| Tile 操作 | 变换方式 |
| --------- | -------- |
| `tile.load`（>2D） | 保持加载原样，之后插入 `tile.reshape` 为 2D |
| `tile.store`（>2D） | 在存储前插入 `tile.reshape` 恢复为 ND |
| `tile.create`/`tile.full`（>2D） | 直接使用展平的 2D 形状重建 |
| `tile.sum`/`tile.max`/`tile.min`（>2D） | 将 axis 映射为 1（2D 的最后轴） |
| 其他 Tile 操作（>2D） | 替换变量，使用 2D 类型重新创建 |
| 1D/2D Tile 操作 | 不变 |

## 示例

**之前**：

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def main_incore_0(self, x: pl.Tensor[[2, 3, 4], pl.FP32],
                      out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
        x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
        y_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
        out_0 = pl.store(y_tile, [0, 0, 0], out_0)
        return out_0
```

**之后**：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def main_incore_0(self, x: pl.Tensor[[2, 3, 4], pl.FP32],
                      out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
        x_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
        x_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.reshape(x_tile_nd, [6, 4])
        y_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
        y_tile_nd: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.reshape(y_tile, [2, 3, 4])
        out_0 = pl.store(y_tile_nd, [0, 0, 0], out_0)
        return out_0
```

3D Tile `[2, 3, 4]` 被展平为 `[6, 4]`。`tile.load` 保持 ND 形状（与张量对接），`tile.reshape` 在 ND 与 2D 之间转换。`tile.store` 之前，Tile 被恢复为 ND 形状。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

**实现文件**：`src/ir/transforms/flatten_tile_nd_to_2d_pass.cpp`

**Python 绑定**：`python/bindings/modules/passes.cpp`

**测试**：`tests/ut/ir/transforms/test_flatten_tile_nd_to_2d.py`

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| 所需 | SSAForm, IncoreTileOps |
| 产生 | SSAForm, TileOps2D |
| 失效 | — |

## 作用范围

| Tile 维度 | 处理方式 |
| --------- | -------- |
| 1D | 不变 |
| 2D | 不变 |
| 3D+ | 展平为 2D |

仅处理 InCore 类型函数（InCore、AIC、AIV）。Orchestration 和 Opaque 函数原样返回。
