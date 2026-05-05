# ExpandMixedKernel Pass

将混合 InCore 函数展开为独立的 AIC（Cube）+ AIV（Vector）内核，并包装在 Group 函数中。非混合 InCore 函数的 FunctionType 会被转换为 AIC 或 AIV。

## 概述

在 `OutlineIncoreScopes` 和 `ConvertTensorToTileOps` 之后，InCore 函数可能同时包含 Cube 操作（`tile.matmul`、`tile.gemv` 等）和 Vector 操作（`tile.add`、`tile.exp` 等）。部分操作如 `tile.load`、`tile.store`、`tile.move`、`tile.reshape` 根据其 tile 操作数的 MemorySpace 被分类为 Cube 或 Vector。包含两侧操作的函数是**混合 InCore 函数**。硬件要求 Cube 和 Vector 操作在不同的核心类型上运行，因此该 Pass 将它们拆分为：

- **AIC 函数**（`FunctionType::AIC`）— 仅包含 Cube + 共享操作
- **AIV 函数**（`FunctionType::AIV`）— 仅包含 Vector + 共享操作
- **Group 函数**（`FunctionType::Group`）— 依次调用 AIC 和 AIV，替换原始函数

当已有 Group 函数调用该 InCore 函数时（如来自 `OutlineClusterScopes`），该 Pass 会**就地改写该 Group 函数**以直接调用 AIC + AIV，避免产生冗余的 Group 包装。若 standalone `Spmd` 包装函数调用该 InCore，则该 Pass 会保留 `Spmd` 包装，并在其下方创建新的 `Group` 被调函数，使 launch 语义继续保留在 `FunctionType::Spmd` 上。

对于**非混合 InCore 函数**（纯 Cube 或纯 Vector），该 Pass 将 `FunctionType::InCore` 转换为对应的类型，无需拆分：

- 纯 Cube → `FunctionType::AIC`
- 纯 Vector 或仅包含共享操作 → `FunctionType::AIV`

该 Pass 执行后，程序中不再存在 `FunctionType::InCore` 函数。

CV 边界的跨核心数据传输通过将显式 `tile.move` 操作拆分为 `tpush`/`tpop` 对来处理：

| 方向 | AIC 侧 | AIV 侧 |
| ---- | ------ | ------ |
| Cube→Vector（如 Acc→Vec） | `tpush_to_aiv(source_tile)` | `dest_var = tpop_from_aic()` |
| Vector→Cube（如 Vec→Mat/Left/Right） | `dest_var = tpop_from_aiv()` | `tile.move` 适配 fractal 布局，然后 `tpush_to_aic(adapted_tile)` |

**Fractal TileView 布局**：跨核传输 tile 的 TileView 由 `BuildCrossCoreTransferView` 根据目标内存空间计算，不同后端的映射不同：

Ascend950（a5）——硬件跨核 pipe 直接按 fractal 布局传输数据：

| 方向 | Push/Pop TileView (blayout, slayout) | 名称 |
| ---- | ------------------------------------ | ---- |
| Vec->Left | col_major, row_major | NZ |
| Vec->Right | row_major, col_major | ZN |
| Vec->Mat | 需要显式在move中设置，否则同src | — |
| Mat/Acc->Vec | 需要显式在move中设置，否则同src | — |

Ascend910B（a2a3）——跨核传输经过 GM → Mat，Mat 仅支持 NZ 布局。因此 Left 和 Right 目标在传输边界统一使用 NZ，最终的 Left/Right 布局由后续 `Mat → Left/Right` `tile.move`（MTE1）处理：

| 方向 | Push/Pop TileView (blayout, slayout) | 名称 |
| ---- | ------------------------------------ | ---- |
| Vec->Left | col_major, row_major | NZ |
| Vec->Right | col_major, row_major | NZ |
| Vec->Mat | 保持原始布局 | — |
| Mat/Acc->Vec | 保持原始布局 | — |

在两种后端上，AIV 推送侧（V→C）都会在 `tpush_to_aic` 前插入一个 `tile.move` 将源 tile 转换为所需的 fractal 布局。`tile.move` 辅助函数（`CreateMove`）在结果类型携带 TileView 时会传播 `blayout`/`slayout` kwargs。

当拆分后的内核包含跨核 `tpush`/`tpop` 时，该 Pass 还会自动在函数前缀补齐前端 pipe setup：

- 消费侧插入 `system.reserve_buffer(...)`
- 生产侧插入 `system.import_peer_buffer(...)`
- 两侧分别插入 `system.aic_initialize_pipe(...)` / `system.aiv_initialize_pipe(...)`

此外，Pass 还会在每条消费侧 `tpop` 链后插入
`system.tfree_to_aic(...)` / `system.tfree_to_aiv(...)`。

这些 setup 参数由拆分后的函数体自动推导：

- `dir_mask`：`C2V=1`、`V2C=2`、双向=`3`
- `slot_size`：所有方向中 tile 字节大小的最大值（`shape * dtype bits / 8`）
- `slot_num`：单向为 `8`，双向时每个方向为 `4`
- `buffer_size`：`slot_num * slot_size`
- buffer 名称：`<func>_c2v_slot_buffer` / `<func>_v2c_slot_buffer`
- reserve-buffer 的 `base`：插入时统一使用 `AUTO`，随后由 `AllocateMemoryAddr` 解析成显式地址

当跨核方向使用了不同大小的 tile 时，Pass 会取所有观察到的 tile 字节大小的最大值作为 `initialize_pipe` 的公共 `slot_size`。较小 tile 写入时不会填满整个槽位，但不影响硬件正确性。

对于消费侧跨核 tile，该 Pass 还会保证每个 `tile.tpop_*` 都有匹配的
`system.tfree_*`。若现有 `tfree` 明显过早，Pass 会在不重排彼此独立 `tpop`
链的前提下，将它延后到同一个 block 中更靠后的语句之后。若 AIC 侧的
`tile.tpop_from_aiv` 结果还需要做同侧 `tile.move`（例如 Mat -> Left/Right/Bias），
则生成的 `system.tfree_to_aiv(...)` 会改写为释放 canonical 的 popped tile，并可被延后到这条 carrier 链之后。

对于 Ascend910B（a2a3），该 Pass 也支持**没有函数级 split 模式**的 mixed kernel（`split` 未设置或为
`SplitMode.None`）。此时 Pass 会保留单个 AIV kernel body，并为其打上**双 AIV 派发**标记；后续 lowering
会基于运行时 `subblock_idx` 插入分支：AIV lane 0 执行原始函数体，AIV lane 1 重放跨核握手，并将 tile-producing
replay 路径强制为 `valid_shape=[0, 0]`，同时屏蔽可见的 `tile.store` 写回。这样
`pl.at(level=CORE_GROUP, optimizations=[pl.auto_chunk])` 形式的 no-split mixed kernel 也能保持 AIC/AIV
握手对称，同时让第二个同步 lane 避免真实 DMA/计算工作。

**前置条件**：

- 输入 IR 必须具有 tile 操作（需先运行 `ConvertTensorToTileOps`）
- 输入 IR 必须已提取 InCore 作用域（需先运行 `OutlineIncoreScopes`）
- Tile 操作必须已展平为 2D（需先运行 `FlattenTileNdTo2D`）
- Tile 内存空间必须已推断（需先运行 `InferTileMemorySpace`）
- 跨核 Fractal TileView 分配在 Ascend950 和 Ascend910B 后端均受支持

**使用时机**：在 `InferTileMemorySpace` 之后运行，当 InCore 函数可能同时包含 Cube 和 Vector tile 操作时使用。

> **注意**：该 Pass 已经在默认 tile 优化流水线中启用。调试问题或自定义 PassPipeline 时，也可以通过
> `passes.expand_mixed_kernel()(program)` 显式调用。

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
  识别哪些 InCore 函数已有 Group 调用者，以及哪些调用者仍然需要保留原始函数名。

阶段 2 — 展开每个 InCore 函数 F：
  1. 递归分类所有语句的亲和性（包括循环/条件内部）
  2. 检测 CV 边界移动：跨越 cube↔vector 内存空间的 tile.move 操作
     （记录在独立的 boundary_moves 映射中，而不是作为单独的亲和性枚举值）
  3. 如果不是混合的（没有 CUBE 操作或没有 VECTOR 操作，且没有边界移动）：
     将 FunctionType 转换为 AIC（纯 Cube）或 AIV（纯 Vector / 仅共享操作）
  4. 构建 AIC 函数体：保留 CUBE + SHARED 语句，删除 VECTOR，递归处理 MIXED 循环
     - 对于边界移动（Cube→Vector）：生成 tpush_to_aiv(source_tile)
     - 对于边界移动（Vector→Cube）：生成 dest_var = tpop_from_aiv()，携带 fractal TileView
  5. 构建 AIV 函数体：对称（保留 VECTOR + SHARED，删除 CUBE）
     - 对于边界移动（Cube→Vector）：生成 dest_var = tpop_from_aic()，携带 fractal TileView
     - 对于边界移动（Vector→Cube）：生成 tile.move 适配 fractal 布局，然后 tpush_to_aic(adapted_tile)
  5a. 通过 BuildCrossCoreTransferView 为所有边界 tpop 结果类型和 tpush 前的 tile.move 操作分配 fractal TileView：
      - Ascend950：Left→NZ，Right→ZN，Mat/Vec→保持原始
      - Ascend910B：Left→NZ，Right→NZ（Mat 仅支持 NZ），Mat/Vec→保持原始
  6. 修复两侧函数体中的循环携带状态
     - 删除在当前核心侧无用的 dead iter_args
     - 为保留下来的 iter_args 补回缺失的 init value 定义
     - 当分支局部值被裁剪后，将悬空 yield 改写为 identity yield
     - 将悬空的 tile.store 结果变量（被 AIC 侧拆分裁剪的 SSA 版本）重映射到对应的输出参数
  7. 对两个函数体运行死代码消除（递归进入循环）
  8. 再次归一化循环携带状态，因为 DCE 可能移除仅用于过渡的 SHARED 后续引用，
     使某些 iter_arg 到这一步才变成可删除
  9. 再次运行死代码消除，清理第二次裁剪后暴露出的 init-value 链
 10. 保证每条消费侧 `tpop` 链都有匹配的
      `system.tfree_to_aic` / `system.tfree_to_aiv`，并在需要时把明显过早的 free 延后到同一个 block 内更靠后的位置
 11. 如果拆分后的函数体包含跨核 tile 操作，且尚未带有 setup，
      则推导并 prepend reserve/import/initialize_pipe 前缀
 12. 创建 AIC 函数（无返回值）和 AIV 函数（原始返回值）
     - 对于 Ascend910B 上的 no-split mixed kernel，为生成的 AIV 打上 dual-dispatch 元数据，
       使后续 lowering 在两个 vector lane 上派发同一个 AIV kernel，并将第二个 lane 的 tile replay
       路径改写为 `valid_shape=[0, 0]`
 13. 如果满足以下任一条件，同时创建 Group 函数（调用 AIC 和 AIV）：
     - 仍有非 Group 调用者需要保留原始函数名（`needs_preserved_name`，例如 standalone Spmd 包装）
     - 尚不存在 Group 调用者（`!has_group_caller`）
 14. 如果已存在 Group 调用者且无需保留名称：跳过额外的 Group 包装

阶段 3 — 改写 Group 调用者：
  对于每个调用了已拆分 InCore 的 Group 函数，将 InCore 调用替换为
  AIC 调用 + AIV 调用序列（AIC 用 EvalStmt，AIV 用 AssignStmt）。
```

**亲和性分类**：

| 亲和性 | 操作 | 分类规则 |
| ------ | ---- | -------- |
| CUBE | `tile.matmul`、`tile.matmul_acc`、`tile.matmul_bias`、`tile.gemv`、`tile.gemv_acc`、`tile.gemv_bias`、`tile.batch_matmul`、`tile.batch_matmul_acc` | 始终为 CUBE（按操作名） |
| CUBE 或 VECTOR | `tile.load` | 按 `target_memory` kwarg：cube 侧内存（Mat、Left、Right、Acc、Bias）→ CUBE；Vec → VECTOR |
| CUBE 或 VECTOR | `tile.store`、`tile.reshape` | 按源 tile 的 `memory_space`：cube 侧内存 → CUBE；Vec → VECTOR |
| MIXED | 跨越 cube↔vector 内存的 `tile.move` | 跨核心侧的叶子移动 —— 同时记录在 `boundary_moves` 映射中（见下文） |
| CUBE 或 VECTOR | `tile.move`（同侧） | 按源 tile 的 `memory_space` |
| VECTOR | 所有其他 `tile.*` 操作（`tile.add`、`tile.exp`、`tile.sub` 等） | 始终为 VECTOR（按操作名） |
| SHARED | 非 tile 操作、函数调用、控制流、标量操作 | — |
| MIXED | 包含 CUBE 和 VECTOR 子语句的复合语句 | — |

**CV 边界检测**：当 `tile.move` 的源 tile 内存和目标内存位于不同核心侧时，该移动为 CV 边界。Cube 侧内存：Mat、Left、Right、Acc、Bias。Vector 侧内存：Vec。同侧移动（如 Mat→Left）按其源内存照常分类。边界叶子移动在亲和性上被标记为 `MIXED`，并额外记录在独立的 `boundary_moves` 映射中；跨核方向（Cube→Vector vs Vector→Cube）由 `CollectCVBoundaryMoves`、`BuildCoreBody` 等调用点通过 `ClassifyMoveDirection` 即时恢复。

**嵌套结构处理**：包含混合操作的 ForStmt、IfStmt 和 WhileStmt 会被复制到 AIC 和 AIV 函数体中，内部内容递归裁剪。

**拆分后的循环状态修复**：在构建 AIC/AIV 函数体时，Pass 会先保留共享的控制流骨架，因此某一侧可能暂时留下多余的 iter_args、缺失的 init value 定义，或引用已被裁剪分支局部值的 yield。Pass 会先在 DCE 前按固定顺序修复这些情况，再在 DCE 后做一次循环状态归一化，因为某些仅用于过渡的共享别名会在 DCE 后消失，进而让相应 iter_arg 变成真正可删除。最后再运行一次 DCE，清理第二次裁剪后暴露出的 init-value 链。

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

**之后**（概念性 — 实际 IR 包含所有变量的类型注解；为简洁起见，这里省略自动生成的 pipe setup 前缀）：

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
        pl.tile.tpush_to_aiv(z_tile, split=0)        # 边界移动：推送 Acc tile 到 AIV

    @pl.function(type=pl.FunctionType.AIV)
    def compute_incore_0_aiv(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        # tpop 结果携带 fractal TileView（Vec 目标 → 保持原始布局）
        z_vec: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView()] = pl.tile.tpop_from_aic(split=0)
        out_0 = pl.store(z_vec, [0, 0], out_0)           # VECTOR 操作
        pl.system.tfree_to_aic(z_vec)                    # 释放已消费的跨核槽位
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

当 `OutlineClusterScopes` 已经创建了调用 InCore 的 Group 函数时，该 Pass 改写已有 Group，而不创建新的包装函数。若 `OutlineClusterScopes` 创建的是 standalone `Spmd` 包装，则该 Pass 会保留该 `Spmd` 函数，并把它改为调用新的同名 `Group` 函数。

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

`MixedKernelExpandedPropertyVerifier` 会检查：

- 剩余的 `FunctionType::InCore` 函数不再同时包含 Cube 和 Vector tile 操作
- AIC 中的 `tile.tpop_from_aiv` 必须落到 `MemorySpace::Mat`
- AIV 中的 `tile.tpop_from_aic` 必须落到 `MemorySpace::Vec`
- 带跨核 `tpush`/`tpop` 的 AIC/AIV 函数必须同时具备所需的 pipe setup
- 每个 AIC/AIV `tile.tpop_*` 都必须有匹配的 `system.tfree_*`
- 跨核 tile 操作的 tile 形状必须是静态已知的（自动 pipe setup 所需）

这样，常见错误（缺少 `initialize_pipe`、缺少 `reserve_buffer` / `import_peer_buffer`、缺少 `tfree`、
非静态 tile 形状、跨核 `tpop`/`tfree` 不匹配）会在 `ExpandMixedKernel` 之后立即报出，而不是拖到 PTO codegen /
`ptoas` 阶段。当前 verifier 只检查按 tile 值/op 的配对关系，并不证明 `tfree` 一定落在该 tile 的真实最后一次使用之后。

## 设计决策

| 决策 | 理由 |
| ---- | ---- |
| 基于 move 的 CV 边界检测 | 显式 `tile.move` 操作标记边界——无需脆弱的变量数据流分析 |
| 使用 `boundary_moves` 映射（而非独立枚举值） | 边界状态可由 `ClassifyMoveDirection` 在调用点即时推导；以侧表存储可让 `CoreAffinity` 保持四个执行侧分类（CUBE / VECTOR / SHARED / MIXED） |
| 数据移动操作基于 MemorySpace 分类 | `tile.load`/`tile.store`/`tile.move`/`tile.reshape` 根据其操作的内存空间服务于 Cube 或 Vector；`InferTileMemorySpace` 在该 Pass 之前已设置此信息 |
| tpop 结果携带 Fractal TileView | tpop 结果类型直接携带 fractal TileView（NZ/ZN），而非剥离布局——下游 Pass 和 codegen 无需额外推断即可看到正确布局 |
| AIV 侧 tpush 前插入 `tile.move` | V→C 传输要求数据为 fractal 布局；在 `tpush_to_aic` 前插入显式 `tile.move` 使布局转换在 IR 中可见 |
| `CreateMove` 传播布局 kwargs | 当结果类型携带 TileView 时，`blayout`/`slayout` 作为 kwargs 转发，使生成的 `tile.move` 调用自描述 |
| Group 保留原始函数名 | 无已有 Group 调用者时：Orchestration 调用点无需修改——不需要重写调用点 |
| 改写已有 Group 调用者 | 当 Group 已调用 InCore 时（如来自 `OutlineClusterScopes`）：就地改写以调用 AIC + AIV，避免冗余的 Group→Group 嵌套 |
| 保留 standalone Spmd 包装 | 当 standalone `Spmd` 调用 InCore 时：保留 `FunctionType::Spmd`，在其下创建 `Group` 被调函数，并继续由 Spmd 持有 `core_num` / `sync_start` |
| 参数复制到所有三个函数 | 简化连接；DCE 在下游 Pass 中移除未使用的参数 |
| 递归处理复合语句 | 正确拆分 `ForStmt`、`IfStmt`、`WhileStmt` 内部的混合操作 |
| 两阶段拆分后循环状态修复 | 先保证 loop-carried state 合法，再在 DCE 移除死共享别名后重新裁剪 iter_arg，最后再跑一次 DCE 清理暴露出的 init-value 链 |
| 自动生成 pipe setup | tensor 级 mixed kernel 无需手写 `reserve_buffer` / `import_peer_buffer` / `initialize_pipe`；Pass 会根据跨核 tile 操作自动推导 |
| 自动生成 tfree 链 | 消费侧拆分内核会补齐缺失的 `tfree`、将其改写为释放 canonical 的 popped tile，并在需要时把明显过早的 free 延后到同一个 block 内更靠后的位置；但不会重排彼此独立的 `tpop` 链顺序 |
| Max-slot-size 策略 | 取所有 tile 字节大小的最大值作为单一 `initialize_pipe.slot_size`，对齐后端"每函数单 reserve/import buffer"的假设，同时支持不同方向混合 tile 大小 |
