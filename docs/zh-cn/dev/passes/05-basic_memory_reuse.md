# BasicMemoryReuse Pass

利用依赖分析识别内存复用机会，并移除冗余的 alloc 操作。

## 概述

该 Pass 通过分析变量生命周期和依赖关系来实现内存共享。在同一内存空间中，生命周期不重叠的变量可以共享内存引用 (MemRef) 对象，从而减少内存占用。

应用 MemRef 共享后，该 Pass 还会**移除冗余的 `block.alloc` 语句 (Statement)**——即那些不再被任何 TileType 变量引用的 MemRef 对应的 alloc 语句。

**核心要点**：

- 生命周期不重叠的变量可以复用内存
- 只有在同一内存空间中的变量才能共享 MemRef
- 生命周期通过 def-use 分析确定
- 共享完成后，已无引用的 MemRef 及其 alloc 语句会被清理

**使用时机**：在 InitMemRef 之后、AllocateMemoryAddr 之前运行。可减少内存分配开销。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::BasicMemoryReuse()` | `passes.basic_memory_reuse()` | 函数级 |

**工厂函数**：

```cpp
Pass BasicMemoryReuse();
```

**Python 用法**：

```python
from pypto.pypto_core import passes

reuse_pass = passes.basic_memory_reuse()
program_optimized = reuse_pass(program)
```

## 算法

1. **依赖图**：使用 `DependencyAnalyzer` 从数据流构建依赖图
2. **生命周期分析**：计算每个变量的 def-use 链和活跃区间
3. **干涉检查**：识别生命周期重叠的变量
4. **MemRef 共享**：为同一内存空间中不干涉的变量分配相同的 MemRef 指针
5. **移除冗余 alloc**：收集仍被 TileType 变量引用的所有 MemRef，然后移除不再使用的 `block.alloc` 语句

**复用条件**：

- 生命周期不重叠（无干涉）
- 相同内存空间
- 大小兼容（复用目标必须足够大）

**Alloc 清理**：

MemRef 共享完成后，部分 MemRef 对象变为无引用状态（其变量现在指向不同的共享 MemRef）。该 Pass 遍历 OpStmts 块，移除所有左值 MemRef 指针不在仍使用集合中的 `block.alloc` `AssignStmt`。空的 OpStmts 块会被完全移除。

## 示例

### MemRef 共享与 Alloc 清理

**之前**（InitMemRef 之后）：

```python
# OpStmts [
mem_vec_0: MemRefType = block.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = block.alloc(Vec, -1, 16384, 1)
mem_vec_2: MemRefType = block.alloc(Vec, -1, 16384, 2)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = block.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = block.add(tile_a, ...)
# tile_a last use ↑
tile_c: Tile[[64, 64], FP32, memref=mem_vec_2] = block.load(...)
# ]
```

**之后**（tile_c 复用了 tile_a 的 mem_vec_0，mem_vec_2 的 alloc 被移除）：

```python
# OpStmts [
mem_vec_0: MemRefType = block.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = block.alloc(Vec, -1, 16384, 1)
# mem_vec_2 alloc removed — no longer referenced
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = block.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = block.add(tile_a, ...)
tile_c: Tile[[64, 64], FP32, memref=mem_vec_0] = block.load(...)
# tile_c now shares mem_vec_0 with tile_a
# ]
```

### 生命周期重叠（不可复用）

**之前/之后**（无变化——alloc 语句保留）：

```python
# OpStmts [
mem_vec_0: MemRefType = block.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = block.alloc(Vec, -1, 16384, 1)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = block.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = block.load(...)
tile_c: Tile[[64, 64], FP32, memref=...] = block.add(tile_a, tile_b)
# tile_a and tile_b are both live here → cannot reuse
# ]
```

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass BasicMemoryReuse();
```

**实现文件**：`src/ir/transforms/basic_memory_reuse_pass.cpp`

- `DependencyAnalyzer` 构建依赖图
- `ComputeLifetimesFromDependencies` 计算活跃区间
- `IdentifyReuseOpportunities` 查找复用候选
- `ApplyMemRefSharing` 通过 `MemRefSharingMutator` 更新 MemRef 指针
- `UsedMemRefCollector` 收集共享后仍被引用的 MemRef 指针
- `RemoveUnusedAllocStatements` 从 OpStmts 中过滤掉冗余的 `block.alloc` 语句

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("basic_memory_reuse", &pass::BasicMemoryReuse, "Memory reuse optimization");
```

**测试**：`tests/ut/ir/transforms/test_basic_memory_reuse.py`

- 测试非重叠生命周期的 MemRef 共享复用
- 测试重叠生命周期不复用
- 测试内存空间隔离
- 测试大小兼容性
- 测试视图操作（reshape）的 MemRef 共享保持
- 测试冗余 alloc 语句移除
