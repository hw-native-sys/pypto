# AllocateMemoryAddr Pass

为已有的 alloc 操作分配实际内存地址。

## 概述

该 Pass 为非 DDR 的内存引用 (MemRef) 分配具体内存地址，并原地更新已有的 `tile.alloc` 语句 (Statement)。它还会在 PTO codegen 之前把 `system.reserve_buffer(base=AUTO)` 解析成显式地址。与创建新的 alloc 操作不同，该 Pass 仅修改由 InitMemRef 创建的 alloc 语句中的地址字段（原值为 `addr=-1`）。

**核心职责**：

- 从 TileType 变量中收集唯一的 MemRef 对象
- 在每个函数中把 `system.reserve_buffer` 的 base 解析成显式地址
- 在每个内存空间内分配顺序的、32 字节对齐的地址
- 更新所有变量类型 (Type) 中的 MemRef 地址
- 使用分配的地址更新 `tile.alloc` 语句参数

**使用时机**：在 MemoryReuse 之后（以尊重共享的 MemRef）、代码生成 (CodeGen) 之前运行。内存管理流水线中的最终 Pass。

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
3. **解析 reserve_buffer**：在每个函数中扫描 `system.reserve_buffer`，为 AUTO buffer 分配显式 base，并计算每个内存空间的保留区末尾地址
4. **分配地址**：对于每个内存空间，委托给 `MemoryAllocatorPolicy` 进行空间过滤、MemRef 排序和地址对齐。默认策略按 ID 排序、使用 32 字节对齐，并从保留区末尾（或 `0`）开始分配
5. **原地更新**：使用 `MemRefUpdateMutator` 完成以下操作：
   - 将变量类型（TileType/TensorType）中的旧 MemRef 引用替换为包含实际地址的新 MemRef
   - 更新已有的 `tile.alloc` `AssignStmt`：替换左值 MemRef 并更新 Call 表达式 (Expression) 中的 addr 参数
   - 把 `system.reserve_buffer` 的 kwargs 改写为显式 `base`

**地址分配（默认策略）**：

- 每个内存空间有独立的地址空间；如果该空间前面已有 `system.reserve_buffer` 保留窗口，则 tile 会从该窗口之后开始分配
- 地址 32 字节对齐：`next_addr = align32(current_addr + size)`
- MemRef 按 ID 排序以确保确定性的分配顺序
- DDR MemRef 被跳过（地址由外部管理）

后端可以通过 `Backend::CreateMemoryAllocatorPolicy()` 提供自定义 `MemoryAllocatorPolicy` 来覆盖上述默认行为。详见下方[分配策略](#分配策略)章节。

## 示例

### 之前（InitMemRef + MemoryReuse 之后）

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 16384, 0)   # addr=-1 (unallocated)
mem_vec_1: MemRefType = tile.alloc(Vec, -1, 16384, 1)   # addr=-1 (unallocated)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.add(tile_a, ...)
# ]
```

### 之后（地址已分配）

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, 0, 16384, 0)      # addr=0
mem_vec_1: MemRefType = tile.alloc(Vec, 16384, 16384, 1)   # addr=16384 (aligned)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.add(tile_a, ...)
# ]
```

### 多内存空间

```python
# Before:
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 2048, 0)
mem_left_1: MemRefType = tile.alloc(Left, -1, 2048, 1)
mem_right_2: MemRefType = tile.alloc(Right, -1, 2048, 2)
mem_acc_3: MemRefType = tile.alloc(Acc, -1, 2048, 3)

# After (each space starts from addr=0):
mem_vec_0: MemRefType = tile.alloc(Vec, 0, 2048, 0)
mem_left_1: MemRefType = tile.alloc(Left, 0, 2048, 1)
mem_right_2: MemRefType = tile.alloc(Right, 0, 2048, 2)
mem_acc_3: MemRefType = tile.alloc(Acc, 0, 2048, 3)
```

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass AllocateMemoryAddr();
```

**实现文件**：`src/ir/transforms/allocate_memory_addr_pass.cpp`

- `MemRefCollectorVisitor` 从 TileType 变量中收集唯一的 MemRef
- `AllocateMemoryAddresses` 使用 `MemoryAllocatorPolicy` 在每个内存空间内分配顺序对齐的地址
- `MemRefUpdateMutator` 在一次遍历中同时更新变量类型和 `tile.alloc` 语句参数

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("allocate_memory_addr", &pass::AllocateMemoryAddr,
           "Allocates real memory addresses for existing alloc operations.");
```

**测试**：`tests/ut/ir/transforms/test_allocate_memory_addr_pass.py`

- 测试 32 字节对齐的地址分配
- 测试多 MemRef 分配
- 测试空函数（无 Tile）
- 测试 alloc 语句被前置到函数体顶层 `SeqStmts`
- 测试 MemRef 去重的原始指针唯一性
- 测试无后端配置时的默认策略行为

## 分配策略

该 Pass 将放置决策委托给 `MemoryAllocatorPolicy` 接口 (`include/pypto/ir/memory_allocator_policy.h`)，使分配策略可扩展而无需修改 Pass 本身。

### 接口

```cpp
class MemoryAllocatorPolicy {
 public:
  virtual ~MemoryAllocatorPolicy() = default;
  virtual bool ShouldAllocate(MemorySpace space) const = 0;
  virtual uint64_t AlignAddress(uint64_t addr, MemorySpace space) const = 0;
  virtual void OrderMemRefs(std::vector<MemRefPtr>& refs) const = 0;
};
```

| 方法 | 用途 | 默认行为 |
| ---- | ---- | -------- |
| `ShouldAllocate` | 过滤哪些内存空间需要分配地址 | 跳过 DDR；分配所有片上空间 |
| `AlignAddress` | 对给定空间的原始地址进行对齐 | 32 字节对齐 |
| `OrderMemRefs` | 在分配前对空间内的 MemRef 排序 | 按 `MemRef::id_` 升序 |

### 默认策略

`DefaultMemoryAllocatorPolicy` 保留了原始硬编码行为（跳过 DDR、32 字节对齐、按 ID 排序）。

### 后端覆盖

当后端已配置（`BackendConfig::IsConfigured()`）时，Pass 调用 `Backend::CreateMemoryAllocatorPolicy()` 获取策略。默认的 `Backend` 实现返回 `DefaultMemoryAllocatorPolicy`。自定义后端可以覆盖此虚方法以提供不同的对齐规则、排序策略或空间过滤：

```cpp
class MyBackend : public Backend {
 public:
  MemoryAllocatorPolicyPtr CreateMemoryAllocatorPolicy() const override {
    return std::make_unique<MyCustomPolicy>();
  }
};
```

当未配置后端时（例如在单元测试中），Pass 会自动回退到 `DefaultMemoryAllocatorPolicy`。
