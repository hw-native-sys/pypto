# InitMemRef Pass

为所有变量初始化内存引用 (MemRef)，并创建地址未分配的 alloc 操作。

## 概述

此 Pass 执行三项任务：

1. **规范化语句 (Statement) 结构**（内部调用 NormalizeStmtStructure）
2. **为 TileType 和 TensorType 变量初始化 MemRef**，分配适当的内存空间
3. **为每个非 DDR 的 MemRef 创建 `tile.alloc` 操作**，地址为 `addr=-1`（未分配）

内存空间分配规则：

- **函数参数** → DDR
- **tile.store 返回值** → DDR（特殊处理，返回 TensorType）
- **其他 tile 操作** → 通过 OpRegistry 内存规格解析（参见 `OpMemorySpaceSpec`）
- **非 tile 变量** → DDR（默认）

**需要**：TypeChecked、SSAForm、SplitIncoreOrch、IncoreTileOps。

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

1. **规范化结构**：调用 `NormalizeStmtStructure` 确保 `SeqStmts` 为扁平结构
2. **分析用法**：遍历函数体，确定每个变量的内存空间
3. **初始化 MemRef**：创建 MemRef 对象（addr=-1）并附加到变量类型
4. **收集非 DDR MemRef**：从 TileType 变量中收集不在 DDR 中的唯一 MemRef 对象
5. **创建 alloc 语句**：为每个非 DDR MemRef 创建 `tile.alloc(memspace, -1, size, id)`
6. **前置 alloc**：将 alloc 语句插入到函数体顶层 `SeqStmts` 的开头

## 示例

**变换前**（经过 SSA/tile 操作转换后）：

```python
def main(input_a: Tensor[[64, 64], FP32], output: Tensor[[64, 64], FP32]):
    tile_a: Tile[[64, 64], FP32] = tile.load(input_a, [0, 0], [64, 64])
    tile_b: Tile[[64, 64], FP32] = tile.add(tile_a, tile_a)
    result: Tensor[[64, 64], FP32] = tile.store(tile_b, [0, 0], output)
    return result
```

**变换后**：

```python
def main(
    input_a: Tensor[[64, 64], FP32, MemRef(space=DDR, addr=-1, id=0)],
    output: Tensor[[64, 64], FP32, MemRef(space=DDR, addr=-1, id=1)],
):
    # SeqStmts [
    mem_vec_2: MemRefType = tile.alloc(Vec, -1, 16384, 2)
    mem_vec_3: MemRefType = tile.alloc(Vec, -1, 16384, 3)
    tile_a: Tile[[64, 64], FP32, memref=mem_vec_2] = tile.load(input_a, [0, 0], [64, 64])
    tile_b: Tile[[64, 64], FP32, memref=mem_vec_3] = tile.add(tile_a, tile_a)
    result: Tensor[[64, 64], FP32, memref=mem_ddr_1] = tile.store(tile_b, [0, 0], output)
    #   ReturnStmt [result]
    # ]
```

关键观察：

- `addr=-1` 表示地址尚未分配（稍后由 AllocateMemoryAddr 完成）
- DDR MemRef（参数）不会生成 `tile.alloc` 语句
- `tile.store` 结果与输出张量参数共享 MemRef
- Alloc 语句放置在函数体顶层 `SeqStmts` 的开头

## ForStmt 循环携带变量

ForStmt 有四个循环携带相关变量，遵循特定的 MemRef 共享规则：

| 角色 | 描述 | MemRef 来源 |
| ---- | ---- | ----------- |
| initValue | 首次迭代前的初始值 | 来自产生该值的操作 |
| iter_arg | 循环体内变量 | 继承自 initValue |
| yield value | 每次迭代结束时产出的值 | 来自产生该值的操作（独立分配） |
| return_var | 循环结束后接收最终 yield 值 | 继承自 yield value |

**共享组**：

- 组 A：initValue + iter_arg（共享同一 MemRef）
- 组 B：yield value + return_var（共享同一 MemRef）

组 A 和组 B 可能有不同的 MemRef。yield 与 iter_arg 之间的 MemRef 不一致由后续的 BasicMemoryReuse 解决（必要时插入 `tile.move`）。

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
- 测试为非 DDR MemRef 创建 tile.alloc 语句
- 测试规范化后的 `SeqStmts` 结构
- 测试 tile.store 结果与输出参数共享 MemRef
- 测试 ForStmt 循环携带变量的 MemRef 关系（initValue/iter_arg 共享，yield/return_var 共享）
