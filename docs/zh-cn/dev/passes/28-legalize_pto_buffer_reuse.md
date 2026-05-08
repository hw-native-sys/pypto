# LegalizePTOBufferReuse Pass

拆分 `MemoryReuse` 留下的、其共享 writer 无法用 PTO 兼容的 `alloc_tile` 与 view op 组合表达的 MemRef。

## 概述

通用的 `MemoryReuse` 基于生命周期、内存空间、dtype、shape，以及（对 2D tile 的）`valid_shape` 差异判定复用。PTO codegen 更严格：同一 MemRef 上多个 tile SSA 值可以下放为多条共享同一 MemRef 地址 / 字节偏移的 `pto.alloc_tile`，但只有当这些非 view writer 拥有**完全相同的 `TileBufSignature`**，或它们之间的差异可由既有 PTO view op materialize 时，共享才合法。否则，必须在地址分配前将该 MemRef 拆分到不同的分配中。

本 Pass 检测非法的跨类型共享，并将冒犯的 writer（及其 view 链上的传递使用者）改绑到新的 MemRef。

**"合法"的跨类型共享** 由 `TileBufSignature::IsPTOMaterializable`（`include/pypto/codegen/pto/tile_buf_signature.h`）定义——即既有 PTO view op 能够表达的差异：

- `tile.reshape` —— 相同 `memory_space` / `dtype` / 布局 (layout) / fractal，元素总数相等
- `tile.extract`、`tile.slice`、`tensor.slice` —— 仅 view 的消费者 (consumer)（不引入新存储）
- `tile.fillpad`、`tile.fillpad_inplace` —— 仅 `pad` 的差异
- 带 padding 的 load / 动态 `valid_shape` —— 仅 `valid_shape` 的差异
- `[1, N]` row-major ↔ `[N, 1]` col-major（物理布局相同的零开销 reshape）

其他差异均视为非法，触发 MemRef 拆分。

**使用时机**：在 `MemoryReuse` 与 `AllocateMemoryAddr` 之间运行。对合法 IR 幂等。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::LegalizePTOBufferReuse()` | `passes.legalize_pto_buffer_reuse()` | 函数级 |

**工厂函数**：

```cpp
Pass LegalizePTOBufferReuse();
```

**Python 用法**：

```python
from pypto.pypto_core import passes

legalize_pass = passes.legalize_pto_buffer_reuse()
program_legal = legalize_pass(program)
```

## Pass Properties

| 属性 | 取值 |
| ---- | ---- |
| Required | `SplitIncoreOrch`、`IncoreTileOps`、`HasMemRefs`、`TileOps2D` |
| Produced | — |
| Invalidated | — |

本 Pass 只是改绑变量并插入新的 `tile.alloc` 语句，不会改变控制流形状、SSA form、或归一化结构。

## 算法

变换以四个阶段在每个 `Function` 上执行：

1. **收集 (`MemRefUsageCollector`)** —— 访问每个定义 tile 类型变量的 `AssignStmt`。对每个 MemRef base 指针记录：
   - **Writers**：非 view 的产生者（如 `tile.load`、`tile.add`、`tile.tpop_from_aic`），其 `TileBufSignature` 仅由 LHS 的 `TileType` 提取（`TileBufSignature::FromTileType`）；RHS call 的输入 `Var` 列表则单独收集，用于下游分析（例如 Ascend910B `load + tpop_from_aic` hazard 检测），并不计入签名本身。
   - **View users**：RHS 是合法 view op 且其源参数与目标位于同一 MemRef 的赋值。Pass 还在 `view_edges` 中记录 source→user 的边，以便后续传递性地转向。
   - **`tile.tpop_from_aic` 集合**：单独追踪，用于下文的 Ascend910B 硬件 hazard。

2. **规划 (`PlanMemRefSplits`)** —— 对每个 writer 数大于 1 的 MemRef：

   1. 取 writer 0 的签名作为参考。
   2. 将其余 writer 按 `IsPTOMaterializable` 与某个已存在组的代表 (representative) 进行兼容性分组：组 0 保留原 MemRef，组 `g ≥ 1` 各自获得新的 MemRef。
   3. 命中 Ascend910B split-AIV `load + tpop_from_aic` hazard（详见下文）的 writer **强制拆分** 至独立组。
   4. 对每个非 0 组，使用 `BuildBasePtrName(memory_space, next_id++)` 为新 base 指针 `Var` 命名，按已观察到的最大分配大小构造新 `MemRef`，并通过 `PropagateSplitToViewUsers` 沿 `view_edges` 将每一个传递性 view user 改绑到新 MemRef。

   `next_id` 由 `MaxMemRefIdCollector` 提取已有 `mem_<space>_<n>` 计数的最大值后递增，避免新生成的名字冲突。

3. **变换 (`MemRefSplitMutator`)** —— 克隆所有受影响的 `Var` / `IterArg`，使其新 `TileType` 指向拆分后的 `MemRef`。所有对旧 `Var` 的引用都通过 `var_remap_` 重映射，使 SSA 用户跟随这次改绑。

4. **插入 alloc (`InsertNewAllocStatements`)** —— 对每个唯一的新 base 指针，使用 `CreateAllocStatement(memref, memory_space)` 构造一条 `tile.alloc` `AssignStmt`。当函数体本身已经是非空的 `SeqStmts` 时，Pass 会将这些新 alloc 前插到该 `SeqStmts` 开头，确保它们出现在使用新 MemRef 的任何用户之前；否则直接返回原 body 不作改动。在 `Default` 流水线中，这一前提由上游建立 `MemoryReuse` 所要求的 `NormalizedStmtStructure` 属性的 pass 保证。

### Ascend910B split-AIV `load + tpop_from_aic` hazard

在 `SplitMode` 非 `None` 的 Ascend910B AIV 函数中，让 (a) `tile.load`（或其任何 view 后代）的输出与 (b) 同时消费 `tile.tpop_from_aic` 值的某个 op 的输入共享同一个 MemRef，会触发硬件 hazard。`BackendHandler::RequiresSplitLoadTpopWorkaround()` 在该后端返回 true；启用时，`CollectForcedSplitWriterIndices` 会标记每一个冒犯的 writer 强制拆分到独立组，无视签名兼容性。

这是本 Pass 中唯一引入后端分派的位置，并且通过 `PassContext::Current()->GetBackendHandler()` 进行——遵循 `.claude/rules/pass-context-config.md`。

## 示例

### 形状不同 → 拆分

两个共享 `mem_vec_0` 的 writer 物理 shape 不同（`[128, 128]` 与 `[64, 64]`），二者互相不可 PTO-materialize，因此第二个 writer 改绑到新的 `mem_vec_1`。

**之前**（`MemoryReuse` 之后）：

```python
# SeqStmts [
mem_vec_0: pl.Ptr = tile.alloc(Vec, 65536)
t1: Tile[[128, 128], FP32, memref=mem_vec_0, Mem.Vec] = tile.load(a, ...)
t2: Tile[[64, 64],  FP32, memref=mem_vec_0, Mem.Vec] = tile.load(a, ...)
result: Tensor[[128, 128], FP32] = tile.store(t2, [0, 0], b)
# ]
```

**之后**：

```python
# SeqStmts [
mem_vec_0: pl.Ptr = tile.alloc(Vec, 65536)
mem_vec_1: pl.Ptr = tile.alloc(Vec, 65536)  # 为 t2 新增
t1: Tile[[128, 128], FP32, memref=mem_vec_0, Mem.Vec] = tile.load(a, ...)
t2: Tile[[64, 64],  FP32, memref=mem_vec_1, Mem.Vec] = tile.load(a, ...)
result: Tensor[[128, 128], FP32] = tile.store(t2, [0, 0], b)
# ]
```

参见 `tests/ut/ir/transforms/test_legalize_pto_buffer_reuse.py::TestIllegalSharingSplit::test_different_shape_same_memref_splits`。

### View 链跟随拆分

被拆分 writer 的 `tile.fillpad` view 必须改绑到新 MemRef，使 view 的存储与其源一致。

**之前**：

```python
t1: Tile[[128, 128], FP32, memref=mem_vec_0, Mem.Vec] = tile.load(a, ...)
t2: Tile[[64, 64],  FP32, memref=mem_vec_0, Mem.Vec] = tile.load(a, ...)
t3: Tile[[64, 64],  FP32, memref=mem_vec_0, Mem.Vec, view(pad=max)]
   = tile.fillpad(t2, pad_value=max)
```

**之后**（`t2` 与其 view `t3` 一同迁移到 `mem_vec_1`）：

```python
mem_vec_1: pl.Ptr = tile.alloc(Vec, 65536)
t1: Tile[[128, 128], FP32, memref=mem_vec_0, Mem.Vec] = tile.load(a, ...)
t2: Tile[[64, 64],  FP32, memref=mem_vec_1, Mem.Vec] = tile.load(a, ...)
t3: Tile[[64, 64],  FP32, memref=mem_vec_1, Mem.Vec, view(pad=max)]
   = tile.fillpad(t2, pad_value=max)
```

参见 `TestIllegalSharingSplit::test_split_propagates_through_view_chain`。

### 合法共享保留

具有**相同** `TileBufSignature`（或仅在 `tile.fillpad` / `tile.reshape` / `valid_shape` 等可 view 实现的差异内不同）的两个 writer 保持原有 MemRef 共享不变。参见 `TestLegalSharingPreserved`。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass LegalizePTOBufferReuse();
```

**实现文件**：`src/ir/transforms/legalize_pto_buffer_reuse_pass.cpp`

- `IsLegalViewOp` —— view op 名称白名单（`tile.reshape`、`tile.extract`、`tile.slice`、`tile.fillpad`、`tile.fillpad_inplace`、`tensor.slice`）
- `MemRefUsageCollector` —— 阶段 1：按 MemRef 收集 writer / view-user / `tile.tpop_from_aic` 索引
- `CollectLoadFamilyVars` / `CollectForcedSplitWriterIndices` —— Ascend910B split-AIV hazard 检测
- `PlanMemRefSplits` —— 阶段 2：签名分组与新 MemRef 分配
- `PropagateSplitToViewUsers` —— 阶段 2 辅助：在 `view_edges` 上 BFS，传递性改绑 view
- `MemRefSplitMutator` —— 阶段 3：用新 MemRef 重写 `Var` / `IterArg` 的类型
- `InsertNewAllocStatements` —— 阶段 4：为每个新 MemRef 在函数体前部插入 `tile.alloc`
- `MaxMemRefIdCollector` —— 由现有名字推断新 id 计数器起点

**后端分派**：`BackendHandler::RequiresSplitLoadTpopWorkaround()`，通过 `PassContext::Current()->GetBackendHandler()` 访问（遵循 `.claude/rules/pass-context-config.md`）。

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("legalize_pto_buffer_reuse", &pass::LegalizePTOBufferReuse,
           "Create a PTO buffer reuse legalisation pass\n\n"
           "After generic MemoryReuse, detects illegal cross-type MemRef sharing\n"
           "that PTO codegen cannot express and splits such MemRefs.");
```

**类型存根 (Type stub)**：`python/pypto/pypto_core/passes.pyi`

```python
def legalize_pto_buffer_reuse() -> Pass:
    """Create a PTO buffer reuse legalisation pass."""
```

**测试**：`tests/ut/ir/transforms/test_legalize_pto_buffer_reuse.py`

- `TestLegalSharingPreserved` —— 相同签名与 `tile.fillpad` view 共享被保留
- `TestAscend910BSplitLoadTpopHazard` —— 910B 上 split-AIV hazard 触发强制拆分；Ascend950 上不强制
- `TestIllegalSharingSplit` —— 不同 shape 拆分与 view 链传递
- `TestLegalizeWithCodegen` —— 通过 PTO codegen 端到端校验 alloc 数量 / 地址
