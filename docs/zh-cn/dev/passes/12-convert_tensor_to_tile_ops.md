# ConvertTensorToTileOps Pass

将 InCore 函数中的 tensor 操作（张量操作）转换为 tile 操作（块操作），并更新编排函数的调用点。

## 概述

`OutlineIncoreScopes` 将 InCore 作用域提取为独立函数后，这些函数仍使用 `TensorType` 变量和 `tensor.*` 操作。本 pass 将其降级为直接映射到 PTO-ISA 指令的 `TileType` 变量和 `tile.*` 操作。

本 pass 还会更新编排/不透明函数中的调用点：为 InCore 函数新增的每个输出参数，在调用点插入 `tensor.create`。

**前置条件**：

- 输入 IR 必须为 SSA 形式
- InCore 作用域必须已提取（需先运行 `OutlineIncoreScopes`）
- 语句结构必须已规范化

**使用时机**：在 `OutlineClusterScopes` 之后、`OptimizeOrchTensors` 之前运行。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::ConvertTensorToTileOps()` | `passes.convert_tensor_to_tile_ops()` | Program 级 |

**Python 用法**：

```python
from pypto.pypto_core import passes

convert_pass = passes.convert_tensor_to_tile_ops()
program_tiled = convert_pass(program)
```

## 算法

本 pass 在 Program 级别分三阶段执行：

### 阶段一：转换 InCore 函数

对每个 `FunctionType::InCore` 函数：

1. **预扫描 MatmulSlice 模式**：收集被 `tensor.matmul` / `tensor.matmul_acc` 使用的 `tensor.slice` 结果。这些需要生成 `tile.load(Mat, transpose=...)` 而非默认的 `tile.load(Vec)`。

2. **插入 tile.load（入口加载）**：为每个被转换 op 直接使用的 `TensorType` 参数，在函数入口插入 `tile.load(param, zeros, shape, shape, target_memory=Vec, transpose=False)`。仅被自加载 op（`tensor.slice`、`tensor.matmul`、`tensor.read`、`tensor.write`、`tensor.assemble`）引用的参数不会生成额外加载。

3. **通过 TensorToTileMutator 转换函数体**：遍历函数体，使用 `OpConversionRegistry` 将每个 `tensor.*` 调用转换为对应的 `tile.*` 调用。Mutator 通过控制流传播类型变更（IterArgs、ForStmt/WhileStmt return_vars、IfStmt return_vars）。

4. **插入 tile.store（出口存储）**：对每个从 `TensorType` 转换为 `TileType` 的返回值，添加 `Out` 参数并插入 `tile.store(tile, zeros, out_param)`。如果返回值来自 `tile.assemble` 循环，则将循环重写为直接使用 `tile.store`（转换时 assemble-loop 重写；与 `OptimizeOrchTensors` 模式 3 不同，该模式处理跨函数优化）。

### 阶段二a：通过 Spmd/Group 包装函数转发新增 Out 参数

`OutlineClusterScopes` 产生的 Spmd/Group 包装函数是对其参数到单个内部 InCore
调用的透明 1:1 转发器。当阶段一为该 InCore 被调用者新增 `Out` 参数时，
包装函数必须在自身签名上镜像这些新增参数并通过内部调用转发给被调用者 ——
否则编排层代码生成的 `BuildWrapperReorderedParams` 不变式（每个内部调用的
`Var` 实参都能解析到某个包装函数参数）会被破坏。

对每个 `FunctionType::Spmd` / `FunctionType::Group` 函数：

1. `ForwardedCallFinder` 查找第一个调用转换后 InCore（阶段一新增了至少一个
   `Out` 参数）的调用点。
2. 若找到，则在包装函数签名末尾追加与 InCore 新增参数类型相同（复用
   `name_hint_`）的 `Out` 参数，并由 `WrapperForwardMutator` 重写该内部调用：
   将新变量追加到实参列表、更新调用返回类型为被调用者新的返回类型。包装
   函数体内部**不会**合成 `tensor.create` —— 分配职责保留在调用者侧。
3. 若未找到转发到转换后 InCore 的调用，则包装函数保持不变。

### 阶段二b：更新编排函数调用点

对每个调用了转换后 InCore 函数或阶段二a 吸收了新增 Out 参数的包装函数的
编排 / 不透明函数：

1. 为每个新增的输出参数插入 `tensor.create`
2. 将创建的张量作为额外参数追加到调用中

InCore、Spmd、Group 函数在本阶段被跳过 —— 它们已在阶段一 / 二a 中被改写。

## MatmulSlice 模式

当 `tensor.slice` 的结果被 `tensor.matmul` 或 `tensor.matmul_acc` 使用时，slice 必须生成 Mat 空间的 tile 而非 Vec 空间。本 pass 预扫描此模式，并根据 matmul kwargs 中的转置标志（LHS 使用 `a_trans`，RHS 使用 `b_trans`）生成 `tile.load(Mat, transpose=...)`。

## 示例

**转换前**：

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
        return y

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x)
        return y
```

**转换后**：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def main_incore_0(
        self, x: pl.Tensor[[64], pl.FP32],
        ret0_out: pl.Out[pl.Tensor[[64], pl.FP32]]
    ) -> pl.Tensor[[64], pl.FP32]:
        x_tile: pl.Tile[[64], pl.FP32] = pl.load(x, (0,), (64,))
        y_tile: pl.Tile[[64], pl.FP32] = pl.tile.add(x_tile, x_tile)
        ret0_store: pl.Tensor[[64], pl.FP32] = pl.store(y_tile, (0,), ret0_out)
        return ret0_store

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        ret0_out: pl.Tensor[[64], pl.FP32] = pl.tensor.create((64,), dtype=pl.FP32)
        y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, ret0_out)
        return y
```

关键变更：

- `pl.add(x, x)` → `pl.tile.add(x_tile, x_tile)`（op 转换）
- 入口插入 `tile.load`，出口插入 `tile.store`
- InCore 函数新增 `Out` 参数 `ret0_out`
- 编排函数调用点插入 `tensor.create`

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

**实现**：`src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp`

**Python 绑定**：`python/bindings/modules/passes.cpp`

**测试**：`tests/ut/ir/transforms/test_convert_tensor_to_tile_ops.py`

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| Required | SSAForm, SplitIncoreOrch, NormalizedStmtStructure |
| Produced | SSAForm, IncoreTileOps, NormalizedStmtStructure |
| Invalidated | — |

## 关键组件

| 组件 | 作用 |
| ---- | ---- |
| `TensorArgsInConvertedOpsCollector` | IRVisitor — 识别需要入口加载的 tensor 参数 |
| `MatmulSlicePatternCollector` | IRVisitor — 查找 slice→matmul 模式以生成 Mat 空间加载 |
| `TypePropagatingMutator` | 基类 IRMutator — 通过控制流传播类型变更 |
| `TensorToTileMutator` | IRMutator — 通过 OpConversionRegistry 将 tensor op 转换为 tile op |
| `ForwardedCallFinder` | IRVisitor — 定位包装函数对转换后 InCore 的调用（阶段二a） |
| `WrapperForwardMutator` | IRMutator — 将新增 Out 参数追加到包装函数的内部调用（阶段二a） |
| `CallSiteUpdateMutator` | IRMutator — 在编排函数调用点插入 tensor.create（阶段二b） |
| `IncoreTileOpsVerifier` | IRVisitor — 验证 InCore 函数中不再包含 TensorType 操作 |

## 作用范围

| 函数类型 | 操作 |
| -------- | ---- |
| InCore | 转换（tensor ops → tile ops）；阶段一可能新增 `Out` 参数 |
| Spmd / Group（转发到转换后 InCore） | 签名镜像 InCore 新增的 `Out` 参数，内部调用转发这些参数（阶段二a） |
| Spmd / Group（未转发到转换后 InCore） | 不变 |
| Orchestration / Opaque | 更新调用点 —— 为每个新增 `Out` 参数插入 `tensor.create`（阶段二b） |
