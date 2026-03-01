# AllocateMemoryAddr Pass

为已有的 alloc 操作分配实际内存地址。

## 概述

该 Pass 为非 DDR 的内存引用 (MemRef) 分配具体内存地址，并原地更新已有的 `block.alloc` 语句 (Statement)。与创建新的 alloc 操作不同，该 Pass 仅修改由 InitMemRef 创建的 alloc 语句中的地址字段（原值为 `addr=-1`）。

**核心职责**：

- 从 TileType 变量中收集唯一的 MemRef 对象
- 在每个内存空间内分配顺序的、32 字节对齐的地址
- 更新所有变量类型 (Type) 中的 MemRef 地址
- 使用分配的地址更新 `block.alloc` 语句参数

**使用时机**：在 BasicMemoryReuse 之后（以尊重共享的 MemRef）、代码生成 (CodeGen) 之前运行。内存管理流水线中的最终 Pass。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::AllocateMemoryAddr()` | `passes.allocate_memory_addr()` | 函数级 |

**工厂函数**：

```cpp
Pass AllocateMemoryAddr();
```

**Python 用法**：

```python
from pypto.pypto_core import passes

alloc_pass = passes.allocate_memory_addr()
program_with_addrs = alloc_pass(program)
```

## 算法

1. **收集 MemRef**：遍历函数体，从 TileType 变量中找到所有唯一的 MemRef 对象
2. **按内存空间分组**：按内存空间（Vec、Mat、Left、Right、Acc）组织 MemRef
3. **分配地址**：对于每个内存空间，按 ID 排序 MemRef 并从 0 开始分配顺序的 32 字节对齐地址
4. **原地更新**：使用 `MemRefUpdateMutator` 完成以下操作：
   - 将变量类型（TileType/TensorType）中的旧 MemRef 引用替换为包含实际地址的新 MemRef
   - 更新已有的 `block.alloc` `AssignStmt`：替换左值 MemRef 并更新 Call 表达式 (Expression) 中的 addr 参数

**地址分配**：

- 每个内存空间有独立的地址空间，从 0 开始
- 地址 32 字节对齐：`next_addr = align32(current_addr + size)`
- MemRef 按 ID 排序以确保确定性的分配顺序
- DDR MemRef 被跳过（地址由外部管理）

## 示例

### 之前（InitMemRef + BasicMemoryReuse 之后）

```python
# OpStmts [
mem_vec_0: MemRefType = block.alloc(Vec, -1, 16384, 0)   # addr=-1 (unallocated)
mem_vec_1: MemRefType = block.alloc(Vec, -1, 16384, 1)   # addr=-1 (unallocated)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = block.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = block.add(tile_a, ...)
# ]
```

### 之后（地址已分配）

```python
# OpStmts [
mem_vec_0: MemRefType = block.alloc(Vec, 0, 16384, 0)      # addr=0
mem_vec_1: MemRefType = block.alloc(Vec, 16384, 16384, 1)   # addr=16384 (aligned)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = block.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = block.add(tile_a, ...)
# ]
```

### 多内存空间

```python
# Before:
mem_vec_0: MemRefType = block.alloc(Vec, -1, 2048, 0)
mem_left_1: MemRefType = block.alloc(Left, -1, 2048, 1)
mem_right_2: MemRefType = block.alloc(Right, -1, 2048, 2)
mem_acc_3: MemRefType = block.alloc(Acc, -1, 2048, 3)

# After (each space starts from addr=0):
mem_vec_0: MemRefType = block.alloc(Vec, 0, 2048, 0)
mem_left_1: MemRefType = block.alloc(Left, 0, 2048, 1)
mem_right_2: MemRefType = block.alloc(Right, 0, 2048, 2)
mem_acc_3: MemRefType = block.alloc(Acc, 0, 2048, 3)
```

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass AllocateMemoryAddr();
```

**实现文件**：`src/ir/transforms/allocate_memory_addr_pass.cpp`

- `MemRefCollectorVisitor` 从 TileType 变量中收集唯一的 MemRef
- `AllocateMemoryAddresses` 在每个内存空间内分配顺序对齐的地址
- `MemRefUpdateMutator` 在一次遍历中同时更新变量类型和 `block.alloc` 语句参数
- DDR MemRef 被跳过（无需地址分配）

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("allocate_memory_addr", &pass::AllocateMemoryAddr,
           "Allocates real memory addresses for existing alloc operations.");
```

**测试**：`tests/ut/ir/transforms/test_add_alloc_pass.py`

- 测试 32 字节对齐的地址分配
- 测试多 MemRef 分配
- 测试空函数（无 Tile）
- 测试 alloc 语句位于首个 OpStmts 内
- 测试 MemRef 去重的原始指针唯一性
