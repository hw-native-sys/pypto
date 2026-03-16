# ExpandMixedKernel Pass

将混合 InCore 函数展开为独立的 AIC（Cube）+ AIV（Vector）内核，并包装在 Group 函数中。非混合 InCore 函数的 FunctionType 会被转换为 AIC 或 AIV。

## 概述

在 `OutlineIncoreScopes` 和 `ConvertTensorToTileOps` 之后，InCore 函数可能同时包含 Cube 操作（`tile.matmul`、`tile.gemv` 等）和 Vector 操作（`tile.add`、`tile.exp` 等）。部分操作如 `tile.load`、`tile.store`、`tile.move`、`tile.reshape` 根据其 tile 操作数的 MemorySpace 被分类为 Cube 或 Vector。包含两侧操作的函数是**混合 InCore 函数**。硬件要求 Cube 和 Vector 操作在不同的核心类型上运行，因此该 Pass 将它们拆分为：

- **AIC 函数**（`FunctionType::AIC`）— 仅包含 Cube + 共享操作
- **AIV 函数**（`FunctionType::AIV`）— 仅包含 Vector + 共享操作
- **Group 函数**（`FunctionType::Group`）— 依次调用 AIC 和 AIV，替换原始函数

当已有 Group 函数调用该 InCore 函数时（如来自 `OutlineClusterScopes`），该 Pass 会**就地改写该 Group 函数**以直接调用 AIC + AIV，避免产生冗余的 Group 包装。仅当 InCore 没有已有的 Group 调用者时，才会创建新的 Group 包装函数。

对于**非混合 InCore 函数**（纯 Cube 或纯 Vector），该 Pass 将 `FunctionType::InCore` 转换为对应的类型，无需拆分：

- 纯 Cube → `FunctionType::AIC`
- 纯 Vector 或仅包含共享操作 → `FunctionType::AIV`

该 Pass 执行后，程序中不再存在 `FunctionType::InCore` 函数。

CV 边界的跨核心数据传输通过将显式 `tile.move` 操作拆分为 `tpush`/`tpop` 对来处理：

| 方向 | AIC 侧 | AIV 侧 |
| ---- | ------ | ------ |
| Cube→Vector（如 Acc→Vec） | `tpush_to_aiv(source_tile)` | `dest_var = tpop_from_aic()` |
| Vector→Cube（如 Vec→Mat） | `dest_var = tpop_from_aiv()` | `tpush_to_aic(source_tile)` |

**前置条件**：

- 输入 IR 必须具有 tile 操作（需先运行 `ConvertTensorToTileOps`）
- 输入 IR 必须已提取 InCore 作用域（需先运行 `OutlineIncoreScopes`）
- Tile 操作必须已展平为 2D（需先运行 `FlattenTileNdTo2D`）
- Tile 内存空间必须已推断（需先运行 `InferTileMemorySpace`）

**使用时机**：在 `InferTileMemorySpace` 之后运行，当 InCore 函数可能同时包含 Cube 和 Vector tile 操作时使用。

> **注意**：该 Pass 尚未加入默认流水线——下游 Pass（`InitMemRef`、`MemoryReuse` 等）尚未全面支持跨核 `tpush`/`tpop`。代码生成已支持 AIC/AIV/Group 函数类型。请通过 `passes.expand_mixed_kernel()(program)` 显式调用。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::ExpandMixedKernel()` | `passes.expand_mixed_kernel()` | 程序级 |

**Python 用法**：

```python
from pypto.pypto_core import passes

expand_pass = passes.expand_mixed_kernel()
program_expanded = expand_pass(program)
```

## 算法

```text
阶段 1 — 预扫描：
  识别具有已有 Group 调用者的 InCore 函数。

阶段 2 — 展开每个 InCore 函数 F：
  1. 递归分类所有语句的亲和性（包括循环/条件内部）
  2. 检测 CV 边界移动：跨越 cube↔vector 内存空间的 tile.move 操作
  3. 如果不是混合的（没有 CUBE 操作或没有 VECTOR 操作，且没有 BOUNDARY 移动）：
     将 FunctionType 转换为 AIC（纯 Cube）或 AIV（纯 Vector / 仅共享操作）
  4. 构建 AIC 函数体：保留 CUBE + SHARED 语句，删除 VECTOR，递归处理 MIXED 循环
     - 对于 BOUNDARY（Cube→Vector）：生成 tpush_to_aiv(source_tile)
     - 对于 BOUNDARY（Vector→Cube）：生成 dest_var = tpop_from_aiv()
  5. 构建 AIV 函数体：对称（保留 VECTOR + SHARED，删除 CUBE）
     - 对于 BOUNDARY（Cube→Vector）：生成 dest_var = tpop_from_aic()
     - 对于 BOUNDARY（Vector→Cube）：生成 tpush_to_aic(source_tile)
  6. 对两个函数体运行死代码消除（递归进入循环）
  7. 创建 AIC 函数（无返回值）和 AIV 函数（原始返回值）
  8. 如果没有已有 Group 调用者：同时创建 Group 函数（调用 AIC 和 AIV）

阶段 3 — 改写 Group 调用者：
  对于每个调用了已拆分 InCore 的 Group 函数，将 InCore 调用替换为
  AIC 调用 + AIV 调用序列（AIC 用 EvalStmt，AIV 用 AssignStmt）。
```

**亲和性分类**：

| 亲和性 | 操作 | 分类规则 |
| ------ | ---- | -------- |
| CUBE | `tile.matmul`、`tile.matmul_acc`、`tile.matmul_bias`、`tile.gemv`、`tile.gemv_acc`、`tile.gemv_bias`、`tile.batch_matmul` | 始终为 CUBE（按操作名） |
| CUBE 或 VECTOR | `tile.load` | 按 `target_memory` kwarg：cube 侧内存（Mat、Left、Right、Acc、Bias）→ CUBE；Vec → VECTOR |
| CUBE 或 VECTOR | `tile.store`、`tile.reshape` | 按源 tile 的 `memory_space`：cube 侧内存 → CUBE；Vec → VECTOR |
| BOUNDARY | 跨越 cube↔vector 内存的 `tile.move` | 源和目标位于不同核心侧（见下文） |
| CUBE 或 VECTOR | `tile.move`（同侧） | 按源 tile 的 `memory_space` |
| VECTOR | 所有其他 `tile.*` 操作（`tile.add`、`tile.exp`、`tile.sub` 等） | 始终为 VECTOR（按操作名） |
| SHARED | 非 tile 操作、函数调用、控制流、标量操作 | — |
| MIXED | 包含 CUBE 和 VECTOR 子语句的复合语句 | — |

**CV 边界检测**：当 `tile.move` 的源 tile 内存和目标内存位于不同核心侧时，该移动为 CV 边界。Cube 侧内存：Mat、Left、Right、Acc、Bias。Vector 侧内存：Vec。同侧移动（如 Mat→Left）按其源内存照常分类。

**嵌套结构处理**：包含混合操作的 ForStmt、IfStmt 和 WhileStmt 会被复制到 AIC 和 AIV 函数体中，内部内容递归裁剪。

## 示例 1：InCore 没有已有 Group 调用者

当 Orchestration 直接调用 InCore 时，创建新的 Group 包装函数。

**之前**（经过 `InferTileMemorySpace` 之后）：

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def compute_incore_0(self, x: pl.Tensor[[16, 128], pl.BF16],
                         y: pl.Tensor[[128, 128], pl.BF16],
                         out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]
                         ) -> pl.Tensor[[16, 128], pl.FP32]:
        x_mat: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128], target_memory=pl.Mem.Mat)
        y_mat: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        x_left: pl.Tile[[16, 128], pl.BF16] = pl.move(x_mat, target_memory=pl.Mem.Left)
        y_right: pl.Tile[[128, 128], pl.BF16] = pl.move(y_mat, target_memory=pl.Mem.Right)
        z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_left, y_right)
        z_vec: pl.Tile[[16, 128], pl.FP32] = pl.move(z_tile, target_memory=pl.Mem.Vec)
        out_0 = pl.store(z_vec, [0, 0], out_0)
        return out_0

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, y) -> pl.Tensor[[16, 128], pl.FP32]:
        out_0 = pl.create_tensor([16, 128], dtype=pl.FP32)
        return self.compute_incore_0(x, y, out_0)
```

**之后**（概念性 — 实际 IR 包含所有变量的类型注解）：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.AIC)
    def compute_incore_0_aic(self, x, y, out_0):
        x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.Mem.Mat)  # CUBE：加载到 Mat
        y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.Mem.Mat) # CUBE：加载到 Mat
        x_left = pl.move(x_mat, target_memory=pl.Mem.Left)   # CUBE：Mat→Left（同侧）
        y_right = pl.move(y_mat, target_memory=pl.Mem.Right)  # CUBE：Mat→Right（同侧）
        z_tile = pl.matmul(x_left, y_right)              # CUBE 操作
        pl.tile.tpush_to_aiv(z_tile, aiv_idx=0)        # BOUNDARY：推送 Acc tile 到 AIV

    @pl.function(type=pl.FunctionType.AIV)
    def compute_incore_0_aiv(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        z_vec: pl.Tile[[16, 128], pl.FP32] = pl.tile.tpop_from_aic(aiv_idx=0)  # BOUNDARY：从 AIC 弹出
        out_0 = pl.store(z_vec, [0, 0], out_0)           # VECTOR 操作
        return out_0

    @pl.function(type=pl.FunctionType.Group)
    def compute_incore_0(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        self.compute_incore_0_aic(x, y, out_0)
        result = self.compute_incore_0_aiv(x, y, out_0)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, y) -> pl.Tensor[[16, 128], pl.FP32]:
        out_0 = pl.create_tensor([16, 128], dtype=pl.FP32)
        return self.compute_incore_0(x, y, out_0)  # 调用 Group（同名）
```

## 示例 2：InCore 有已有 Group 调用者

当 `OutlineClusterScopes` 已经创建了调用 InCore 的 Group 函数时，该 Pass 改写已有 Group，而不创建新的包装函数。

**之前**：

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def compute_incore_0(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        # ... 混合 Cube + Vector 操作 ...

    @pl.function(type=pl.FunctionType.Group)
    def compute_group(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        result = self.compute_incore_0(x, y, out_0)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, y) -> pl.Tensor[[16, 128], pl.FP32]:
        out_0 = pl.create_tensor([16, 128], dtype=pl.FP32)
        return self.compute_group(x, y, out_0)
```

**之后** — 已有 Group 被改写，不存在 `compute_incore_0` Group 包装：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.AIC)
    def compute_incore_0_aic(self, x, y, out_0):
        # ... Cube 操作 + tpush ...

    @pl.function(type=pl.FunctionType.AIV)
    def compute_incore_0_aiv(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        # ... tpop + Vector 操作 ...

    @pl.function(type=pl.FunctionType.Group)
    def compute_group(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        self.compute_incore_0_aic(x, y, out_0)       # 改写后：AIC 调用
        result = self.compute_incore_0_aiv(x, y, out_0)  # 改写后：AIV 调用
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, y) -> pl.Tensor[[16, 128], pl.FP32]:
        out_0 = pl.create_tensor([16, 128], dtype=pl.FP32)
        return self.compute_group(x, y, out_0)  # 不变
```

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

**实现文件**：`src/ir/transforms/expand_mixed_kernel_pass.cpp`

**Python 绑定**：`python/bindings/modules/passes.cpp`

**测试**：`tests/ut/ir/transforms/test_expand_mixed_kernel.py`

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| 所需 | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D, TileMemoryInferred |
| 产生 | SSAForm, MixedKernelExpanded |
| 失效 | — |

## 属性验证器

`MixedKernelExpandedPropertyVerifier` 检查剩余的 `FunctionType::InCore` 函数不同时包含 Cube 和 Vector tile 操作。AIC/AIV/Group 函数不做检查（它们已按定义完成拆分）。

## 设计决策

| 决策 | 理由 |
| ---- | ---- |
| 基于 move 的 CV 边界检测 | 显式 `tile.move` 操作标记边界——无需脆弱的变量数据流分析 |
| CV move 使用 BOUNDARY 亲和性 | 将边界处理与 CUBE/VECTOR/MIXED 逻辑清晰分离 |
| 数据移动操作基于 MemorySpace 分类 | `tile.load`/`tile.store`/`tile.move`/`tile.reshape` 根据其操作的内存空间服务于 Cube 或 Vector；`InferTileMemorySpace` 在该 Pass 之前已设置此信息 |
| Group 保留原始函数名 | 无已有 Group 调用者时：Orchestration 调用点无需修改——不需要重写调用点 |
| 改写已有 Group 调用者 | 当 Group 已调用 InCore 时（如来自 `OutlineClusterScopes`）：就地改写以调用 AIC + AIV，避免冗余的 Group→Group 嵌套 |
| 参数复制到所有三个函数 | 简化连接；DCE 在下游 Pass 中移除未使用的参数 |
| 递归处理复合语句 | 正确拆分 `ForStmt`、`IfStmt`、`WhileStmt` 内部的混合操作 |
