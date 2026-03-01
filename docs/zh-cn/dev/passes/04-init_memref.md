# InitMemRef Pass

为所有变量初始化内存引用 (MemRef)，并创建地址未分配的 alloc 操作。

## 概述

此 Pass 执行三项任务：

1. **规范化语句 (Statement) 结构**（内部调用 NormalizeStmtStructure）
2. **为 TileType 和 TensorType 变量初始化 MemRef**，分配适当的内存空间
3. **为每个非 DDR 的 MemRef 创建 `block.alloc` 操作**，地址为 `addr=-1`（未分配）

内存空间分配规则：

- **函数参数** -> DDR
- **block.load/block.move** -> 从 `target_memory` 关键字参数提取（默认 Vec）
- **block.store** -> DDR（与输出张量 (Tensor) 共享 MemRef）
- **block.matmul/block.matmul_acc** -> Acc
- **其他块操作** -> Vec
- **其他变量** -> DDR（默认）

**需要**：TypeChecked、SSAForm、SplitIncoreOrch、IncoreBlockOps。

**产生**：HasMemRefs、NormalizedStmtStructure。

**失效**：SSAForm（引入了新的 MemRef 变量）。

**使用时机**：在静态单赋值 (SSA) 转换、提取和块操作转换之后运行。在 BasicMemoryReuse、InsertSync 和 AllocateMemoryAddr 之前必须运行。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::InitMemRef()` | `passes.init_mem_ref()` | 函数级 |

**工厂函数**：

```cpp
Pass InitMemRef();
```

**Python 用法**：

```python
from pypto.pypto_core import passes

init_pass = passes.init_mem_ref()
program_with_memrefs = init_pass(program)
```

## 算法

1. **规范化结构**：调用 `NormalizeStmtStructure` 确保 SeqStmts/OpStmts 结构
2. **分析用法**：遍历函数体，确定每个变量的内存空间
3. **初始化 MemRef**：创建 MemRef 对象（addr=-1）并附加到变量类型
4. **收集非 DDR MemRef**：从 TileType 变量中收集不在 DDR 中的唯一 MemRef 对象
5. **创建 alloc 语句**：为每个非 DDR MemRef 创建 `block.alloc(memspace, -1, size, id)`
6. **插入到第一个 OpStmts**：将 alloc 语句前置到函数体的第一个 OpStmts 中

## 示例

**变换前**（经过 SSA/块操作转换后）：

```python
def main(input_a: Tensor[[64, 64], FP32], output: Tensor[[64, 64], FP32]):
    tile_a: Tile[[64, 64], FP32] = block.load(input_a, [0, 0], [64, 64])
    tile_b: Tile[[64, 64], FP32] = block.add(tile_a, tile_a)
    result: Tensor[[64, 64], FP32] = block.store(tile_b, [0, 0], [64, 64], output)
    return result
```

**变换后**：

```python
def main(
    input_a: Tensor[[64, 64], FP32, MemRef(space=DDR, addr=-1, id=0)],
    output: Tensor[[64, 64], FP32, MemRef(space=DDR, addr=-1, id=1)],
):
    # SeqStmts [
    #   OpStmts [
    mem_vec_2: MemRefType = block.alloc(Vec, -1, 16384, 2)
    mem_vec_3: MemRefType = block.alloc(Vec, -1, 16384, 3)
    tile_a: Tile[[64, 64], FP32, memref=mem_vec_2] = block.load(input_a, [0, 0], [64, 64])
    tile_b: Tile[[64, 64], FP32, memref=mem_vec_3] = block.add(tile_a, tile_a)
    result: Tensor[[64, 64], FP32, memref=mem_ddr_1] = block.store(tile_b, [0, 0], [64, 64], output)
    #   ]
    #   ReturnStmt [result]
    # ]
```

关键观察：

- `addr=-1` 表示地址尚未分配（稍后由 AllocateMemoryAddr 完成）
- DDR MemRef（参数）不会生成 `block.alloc` 语句
- `block.store` 结果与输出张量参数共享 MemRef
- Alloc 语句放置在第一个 OpStmts 的开头

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass InitMemRef();
```

**实现文件**：`src/ir/transforms/init_memref.cpp`

- `NormalizeStmtStructure` 在 MemRef 初始化之前被内部调用
- `MemRefUsageVisitor` 分析每个变量的内存空间
- `InitMemRefMutator` 创建 MemRef 对象并附加到类型
- `NonDDRMemRefCollector` 收集唯一的非 DDR MemRef
- `CreateAllocStatement` / `InsertAllocsIntoBody` 创建并插入 alloc 操作

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("init_mem_ref", &pass::InitMemRef, "Initialize MemRef for variables");
```

**测试**：`tests/ut/ir/transforms/test_init_memref.py`

- 测试内存空间分配（Vec、Mat、Left、Right、Acc、DDR）
- 测试所有 MemRef 的 addr=-1
- 测试为非 DDR MemRef 创建 block.alloc 语句
- 测试规范化的 SeqStmts/OpStmts 结构
- 测试 block.store 结果与输出参数共享 MemRef
