# MemoryReuse Pass

利用依赖分析识别内存复用机会，并移除冗余的 alloc 操作。

## 概述

该 Pass 通过分析变量生命周期和依赖关系来实现内存共享。在同一内存空间中，生命周期不重叠的变量可以共享内存引用 (MemRef) 对象，从而减少内存占用。

应用 MemRef 共享后，该 Pass 还会**移除冗余的 `tile.alloc` 语句 (Statement)**——即那些不再被任何 TileType 变量引用的 MemRef 对应的 alloc 语句。

**核心要点**：

- 生命周期不重叠的变量可以复用内存
- 只有在同一内存空间中的变量才能共享 MemRef
- 生命周期通过 def-use 分析确定
- 共享完成后，已无引用的 MemRef 及其 alloc 语句会被清理

**使用时机**：在 InitMemRef 之后、AllocateMemoryAddr 之前运行。可减少内存分配开销。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::MemoryReuse()` | `passes.memory_reuse()` | 函数级 |

**工厂函数**：

```cpp
Pass MemoryReuse();
```

**Python 用法**：

```python
from pypto.pypto_core import passes

reuse_pass = passes.memory_reuse()
program_optimized = reuse_pass(program)
```

## 算法

1. **生命周期分析**：遍历完整 IR 树（包括嵌套控制流体内的语句）通过 def-use 分析计算变量生命周期。在循环外定义但在循环内使用的变量，其生命周期会延展到循环结束（循环感知延展）
2. **干涉检查**：识别生命周期重叠的变量
3. **MemRef 共享**：为同一内存空间中不干涉的变量分配相同的 MemRef 指针
4. **循环携带变量重对齐**（`AlignLoopCarriesToInitMutator`）：共享（步骤 3）只会重写由 `AssignStmt` 定义的变量（producer/init），而循环携带的 `iter_arg`/`return_var` 节点被排除在生命周期/共享映射之外、仍保留原始 MemRef。本步骤**自外向内**遍历 `ForStmt`，将每个循环的 `iter_arg`/`return_var` 重对齐到其（已复用的）`initValue` 的 MemRef，并在递归前写入 `var_remap_`，使嵌套循环能观察到已修正的外层 `iter_arg` 作为其 init。若缺少本步骤，被复用的**嵌套流水化 `matmul_acc`** 累加器会分裂到两个 Acc 缓冲区，导致步骤 5 插入非法的 `acc→acc tile.move`，被 Ascend 910B 的 ptoas 拒绝（[#1352](https://github.com/hw-native-sys/pypto/issues/1352)）
5. **Yield 修复**：修复控制流返回变量的 MemRef 不一致：
   - **ForStmt**：确保 4 个循环携带变量（initValue、iter_arg、yield value、return_var）共享同一个 MemRef。若 MemRef 不同则在 yield 前插入 `tile.move`
   - **IfStmt**：修补 return_vars 使其 MemRef 与 yield value 一致
6. **移除冗余 alloc**：收集仍被 TileType 变量引用的所有 MemRef，然后移除不再使用的 `tile.alloc` 语句

**复用条件**：

- 生命周期不重叠（无干涉）。当 `prev.last_use <= curr.def` 时，两个变量不重叠（即源的最后使用可以和目标的定义在同一语句，因为在同一语句内输入先于输出被消费）
- 相同内存空间
- 大小兼容（复用目标必须足够大）
- **L0 cube 输入例外（Left/Right）**：`Mem.Left` / `Mem.Right` 的缓冲区存放的是由 view 算子（`tile.extract` / `tile.slice` / `tile.reshape`）产生的子 tile，PTO codegen 会按每个 tile 变量在缓冲区基址处单独物化。因此同一 L0 空间、生命周期不重叠、**字节**大小足够的两个这类缓冲区，即使 **shape 不同**也可以共享同一槽位 —— 对它们跳过下面的 `AreTileTypesCompatible`（shape/dtype/view）检查（前提是两端的 producer 都是 view 算子，且属于 [`LegalizePtoBufferReuse`](30-legalize_pto_buffer_reuse.md) 的 `IsLegalViewOp` 子集，从而共享的 MemRef 能在该 pass 中存活）。这使 fused-attention 能用 QK 的 `Right` 缓冲区（`[k, SEQ]`）复用 PV 的 `Right` 缓冲区（`[k', HEAD]`），将 L0B 峰值减半（issue #1595）。其它空间（Vec/Acc/Mat）仍保持严格匹配：
- TileType 兼容性 — 由 `AreTileTypesCompatible` 检查：
  - 相同 shape（所有维度必须精确匹配）
  - 相同 dtype（例如 FP32 与 BF16 阻止复用，自动处理 `tile.cast`）
  - 相同 TileView 存储属性：`stride`、`start_offset`、`blayout`、`slayout`、`fractal`、`pad` 必须都结构相等（例如 `tile.fillpad` 改变 `pad`，因此其输出不能复用其输入 —— 仅 `pad` 不一致即阻止复用）
  - 当存在的 view 在存储上是平凡的（trivial）时，view 的**有无**可以不同：无 TileView 的 tile 具有默认物理存储（连续、零偏移、row-major/none-box、默认 fractal、无 pad），因此它与一个仅设置了 `valid_shape`（其余存储字段均为默认值）的 view tile 兼容。这使得复用可以跨越带有非对称 view 的结构克隆 tile，例如 `SplitVectorKernel` 产生的 dual-AIV 派发 `if` 的两个互斥分支 —— 其中一个分支的 tile 带有平凡的 `valid_shape` view，另一个分支的 tile 没有 view。若某个 view 的存储字段偏离默认值，则它仍与无 view 的 tile 不兼容。
  - 对于 2D tile，`valid_shape` 不要求匹配：复用后每个 tile 在自己的 TileType 中保留各自的 `valid_shape`，PTO codegen 会为每个变量发射带有各自静态 valid 范围的 `alloc_tile` 声明，它们共享底层 buffer。这样，`PartialUnrollTileLoops` 产生的仅在边界守护 `valid_shape` 上不同的兄弟分支 tile 可以共用一个后备分配。对于 N-D tile，`valid_shape` 不一致仍然阻止复用。

**Alloc 清理**：

MemRef 共享完成后，部分 MemRef 对象变为无引用状态（其变量现在指向不同的共享 MemRef）。该 Pass 遍历周围的 `SeqStmts`，移除所有左值 MemRef 指针不在仍使用集合中的 `tile.alloc` `AssignStmt`。

## 示例

### MemRef 共享与 Alloc 清理

**之前**（InitMemRef 之后）：

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = tile.alloc(Vec, -1, 16384, 1)
mem_vec_2: MemRefType = tile.alloc(Vec, -1, 16384, 2)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.add(tile_a, ...)
# tile_a last use ↑
tile_c: Tile[[64, 64], FP32, memref=mem_vec_2] = tile.load(...)
# ]
```

**之后**（tile_c 复用了 tile_a 的 mem_vec_0，mem_vec_2 的 alloc 被移除）：

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = tile.alloc(Vec, -1, 16384, 1)
# mem_vec_2 alloc removed — no longer referenced
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.add(tile_a, ...)
tile_c: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
# tile_c now shares mem_vec_0 with tile_a
# ]
```

### 生命周期重叠（不可复用）

**之前/之后**（无变化——alloc 语句保留）：

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = tile.alloc(Vec, -1, 16384, 1)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.load(...)
tile_c: Tile[[64, 64], FP32, memref=...] = tile.add(tile_a, tile_b)
# tile_a and tile_b are both live here → cannot reuse
# ]
```

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass MemoryReuse();
```

**实现文件**：`src/ir/transforms/memory_reuse_pass.cpp`

- `LifetimeAnalyzer` 遍历完整 IR 树计算变量生命周期（包括嵌套控制流）
- `ComputeLifetimes` 构建 MemRef 共享组和生命周期区间
- `IdentifyReuseOpportunities` 查找复用候选
- `ApplyMemRefSharing` 通过 `MemRefSharingMutator` 更新 MemRef 指针
- `YieldFixupMutator` 修复 ForStmt/IfStmt 在复用后的 yield/return_var MemRef 不一致（必要时插入 `tile.move`）
- `UsedMemRefCollector` 收集共享后仍被引用的 MemRef 指针
- `RemoveUnusedAllocStatements` 从 `SeqStmts` 中过滤掉冗余的 `tile.alloc` 语句

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("memory_reuse", &pass::MemoryReuse, "Memory reuse optimization");
```

**测试**：`tests/ut/ir/transforms/test_memory_reuse.py`

- 测试非重叠生命周期的 MemRef 共享复用
- 测试重叠生命周期不复用
- 测试内存空间隔离
- 测试大小兼容性
- 测试切片操作的 MemRef 共享保持
- 测试冗余 alloc 语句移除
- 测试控制流生命周期分析（ForStmt 内嵌套 IfStmt、分支变量共享）
