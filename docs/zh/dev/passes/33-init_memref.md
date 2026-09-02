# InitMemRef Pass

物化编译器负责的 PTO level3 scratch，为所有变量初始化内存引用 (MemRef)，并创建地址未分配的 alloc 操作。

## 概述

此 Pass 执行四项任务：

1. **规范化语句 (Statement) 结构**（内部调用 NormalizeStmtStructure）
2. **物化编译器负责的 level3 scratch**：处理 A2/A3 ABI 的 `tile.ci`、窄化 `tile.cast`，以及 A2/A3 与 A5 上需要 scratch 的 `tile.sort32`；`tile.sel` / `tile.sels` / `tile.prelu` 的 caller tmp 原样保留
3. **为 TileType 和 TensorType 变量初始化 MemRef**
4. **为每个非 DDR 的 MemRef 创建 `tile.alloc` 操作**，地址为 `addr=-1`（未分配）

内存空间从 `TileType::memory_space_` 读取（由 InferTileMemorySpace 设置）。无 `memory_space` 的变量默认为 DDR。

**需要**：SSAForm、SplitIncoreOrch、IncoreTileOps、TileOps2D、TileMemoryInferred。

**产生**：HasMemRefs、NormalizedStmtStructure。

**失效**：SSAForm（引入了新的 MemRef 变量）。

**使用时机**：在静态单赋值 (SSA) 转换、提取和块操作转换之后运行。在 MemoryReuse 和 AllocateMemoryAddr 之前必须运行。

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
2. **物化 level3 scratch**：在 PyPTO 或 DSA-RP 规划模式下，为缺失的编译器 scratch 插入普通 Vec `tile.create`。`tile.ci` 与窄化 `tile.cast` 仍仅适用于 A2/A3，必要的 `tile.sort32` scratch 则同时适用于 A2/A3 和 A5。`tile.sel` / `tile.sels` / `tile.prelu` 的 caller tmp 原样保留。PTOAS 规划器保持不变，由其 level-2 `PlanMemory` 管理隐式 scratch。
3. **解析声明式分配**：收集所有单参数 `pl.MemRef(...)` 声明，并从绑定的 tile 推导出每块分配的大小与内存空间（见[声明式分配](#声明式分配)）
4. **初始化 MemRef**：从 `TileType` 读取 `memory_space`（由 InferTileMemorySpace 设置），创建 MemRef 对象（addr=-1）并附加到变量类型
   - **tile.store**：结果与输出 tensor 参数共享 MemRef（由 `output_reuses_input_arg` 注册表属性指定）
   - **View 操作**（如 `tile.reshape`）：输出与输入 tile 共享 MemRef
   - **复用输入操作**（如 `tile.matmul_acc`、`tile.gemv_acc`）：输出与指定输入共享 MemRef（由 `output_reuses_input_arg` 注册表属性指定）
   - **ForStmt/IfStmt return_vars**：修补为与对应 yield 值共享 MemRef
   - **用户绑定的 tile**：保留作者指定的 buffer，而不是新建一块分配
5. **收集非 DDR MemRef**：从 TileType 变量中收集不在 DDR 中的唯一 MemRef 对象
6. **创建 alloc 语句**：为每个非 DDR MemRef 创建 `tile.alloc(memspace, -1, size, id)`；base 属于声明式分配时带上 `pinned=True`
7. **前置 alloc**：将 alloc 语句插入到函数体顶层 `SeqStmts` 的开头

## 声明式分配

`pl.Tile[[...], dtype, <alloc>, pl.Mem.Vec]` 将一个 tile 绑定到由 kernel 作者声明的分配
上，其中 `<alloc>` 是一个按变量引用的 `pl.MemRef("name")`（或同样的单参数内联形式，也就是
打印器输出的形式）。引用同一块分配的 tile 共享它，且 `MemoryReuse` 绝不会把其他 tile 塞进去。这是手工复用控制——作者为何需要它,见 [MemoryReuse](35-memory_reuse.md#声明式分配)。

**声明如何抵达本 pass。** 解析器把单参数 `pl.MemRef` 解析为一个 `MemRef`，其 `base_` Ptr 按
名字 intern（因此命名同一块分配的两处注解共享同一个 base），`byte_offset = 0`、不带大小，并且
**置上 `is_pinned_`**。这个标志位正是声明的识别依据——重新解析 post-allocation dump 时
同样会在 `TileType` 上出现 MemRef，而那些是编译器的分配。把它显式记录下来（而不是从哨兵大小、或
"pass 站在流水线哪个位置"去推断），使该分类成为数据自身的属性，也让打印器可以把声明输出为
单参数形式——无需编造大小与地址即可完成往返。

本 pass 会**消费**掉这个声明：它产出的 MemRef 是一个携带推导大小的普通 MemRef，标志位已清除。
此后由分配点的 `pinned=True` kwarg 标记该分配归作者所有。

槽位信息是例外——`slot_count_` 与 `slot_index_` 会保留在解析后的 MemRef 上。把下标折算进
`byte_offset_` 回答的是槽位**落在哪里**，并不改变"这是某块 N 槽分配的第 k 个槽位"这一事实；
PTO codegen 正是读取它，才能发出一块 ptoas `pto.alloc_multi_tile` 区域并在每个使用点发一条
`pto.multi_tile_get`，而不是 N 块互不相关的分配。两者是派生关系而非彼此独立：`byte_offset_`
由下标算出，`AllocateMemoryAddr` 还可能把它重定位到物理地址上，因此下标是作者的**选择**，
偏移是它解析后的**位置**。两者都会打印，所以 dump 可以按
`pl.MemRef(base, offset, size, slots=N)[k]` 往返。

在 `memory_planner=PTOAS` 下，**单槽位**声明会被拒绝：它的隔离性由 `MemoryReuse` 保证，而该
pass 被 ptoas 整体替换，ptoas 侧也没有对应概念来接管。多槽位声明则被接受——它的槽位会成为一块
ptoas 区域，ptoas 被禁止合并这些物理段，详见
[Python 语法](../language/00-python_syntax.md#在-ptoas-内存规划器下)。

声明位于被赋值的 `Var` 上而非 RHS `Call` 上——`ConvertToSSA` 只把它合并进 Var 的类型，而算子类型
推导永远不产生 MemRef——因此任何从 Call 重建类型的 pass 都必须显式携带它。`ConvertToSSA` 负责
合并；`FlattenTileNdTo2D` 在四处重建中都携带（ND 展平、≤2D `tile.load`、rank>2 `tile.create`/`tile.full`、通用 tile 算子）；
`InferTileMemorySpace` 在把 Var 类型同步到重建后的 Call 时保留它。只克隆而不重建的 pass（包括
`LowerPipelineLoops` 产出的各流水级 body）经由 `MemRef` 的克隆路径保留它。

**本 pass 推导的内容。** 作者既不写大小也不写地址：

| 属性 | 推导来源 |
| ---- | -------- |
| 大小 | 绑定到该 buffer 的最大 tile |
| 内存空间 | 绑定 tile 共有的空间（必须一致） |
| 地址 | 交给 `AllocateMemoryAddr`，与编译器分配完全一致 |

**被拒绝的绑定**（均为 `pypto::ValueError`，且携带出错 tile 的 span）：

- 绑定的 tile 是动态 shape——声明式分配必须在编译期定型。
- 同一 buffer 上的 tile 内存空间不一致。
- 绑定 view / 原地算子（`tile.reshape`、`tile.matmul_acc` 等）的输出。这类结果物理上**就是**
  其源 tile 的 buffer，无法另行放置；应改为绑定源 tile。

第四条规则——绑定到**同一槽位**的 tile 生命周期不得重叠——需要生命周期信息，因此在
[MemoryReuse](35-memory_reuse.md#声明式分配) 中检查。位于*不同*槽位的 tile 本来就应该同时
存活，这正是多槽位声明的用途。

```python
ping, pong = pl.MemRef(), pl.MemRef()

t0: pl.Tile[[64, 64], pl.FP32, ping, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
t1: pl.Tile[[64, 64], pl.FP32, pong, pl.Mem.Vec] = pl.exp(t0)
t2: pl.Tile[[64, 64], pl.FP32, ping, pl.Mem.Vec] = pl.exp(t1)  # 与 t0 共用
```

会得到两块 pinned 分配，`t0` 与 `t2` 落在 `ping` 上：

```python
ping: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384, pinned=True)
pong: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384, pinned=True)
```

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
- `tile.store` 结果与输出张量参数共享 MemRef（通过 `output_reuses_input_arg` 注册表属性指定）
- 复用输入操作（`tile.store`、`matmul_acc`、`gemv_acc`）与指定输入共享 MemRef，避免冗余 alloc
- Alloc 语句放置在函数体顶层 `SeqStmts` 的开头

## 切片视图遵循父张量的布局

`tile.slice` / `tensor.slice` 的结果与父张量共享 `base_` Ptr，并附加一个字节偏移
（byte offset）和收窄后的视图尺寸。只要该 pass 能够精确描述父张量的真实存储布局，
这两个数值就在**该布局下**计算，而不是固定按行主序稠密（row-major dense）模型计算：

| 父张量布局 | 逻辑坐标 `(r, c)` 的偏移 | 视图跨度 |
| ---------- | ------------------------ | -------- |
| `none_box` 行主序稠密（Vec / DDR，以及所有 `tensor.slice`） | `(r * cols + c) * elem_bytes` | 按父张量步长，从首元素到末元素的包络 |
| 累加器（L0C）：`blayout=col_major`、`slayout=row_major`、`fractal=1024`，**且仅限保留完整行范围的窗口** | `c * rows * elem_bytes` | 恰好等于其块列占用的 `cols * rows * elem_bytes`，且连续 |

L0C 并非行主序稠密：它是由 16x16 分形块（box）构成的网格，**按块列优先**存放，因此
`[M, N]` 累加器中第 `(r_b, c_b)` 个块起始于 `(c_b * M/16 + r_b) * 1024` 字节。
`tile_view_semantics::GetAccumulatorTileGeometry` 从父张量的有效 `TileView` 识别该布局
组合并返回两个元素步长——一个逻辑行步进对应一个块宽（16 个元素），一个逻辑列步进对应
一整个物理列（`M` 个元素）。

即使 `tile.slice` 通常会下降为 `pto.subview`（它携带**逻辑**窗口下标并自行重新推导地址），
这一区分仍然重要：某些视图操作**不会**下降为 subview——例如叠加在切片之上的
`tile.reshape`——它会继承该字节偏移并将其作为自己的 `pto.alloc_tile addr`。此处的行主序
数值不只是未对齐，而是指向了错误的数据。

### 为什么只支持保留完整行范围的窗口

被继承的偏移会成为一个**独立**地址，而包裹它的 `alloc_tile` 携带的是*切片自身*的 `rows`。
PTO 由该 `rows` 推导块列步长，因此只有当窗口保留父张量的完整行范围时，NZ 形式才描述的是
父张量的分块：此时 `rows` 未变，每个块都落在父张量放置它的位置上。

`GetSliceAccumulatorGeometry`（`include/pypto/ir/transforms/utils/memref_utils.h`）是偏移
与跨度共同经过的唯一判定条件。它仅在以下情况接受某个窗口：

- 父张量是二维、静态形状、整块对齐、元素宽度为 4 字节的 NZ 累加器；
- 窗口保留父张量的完整行范围，且起始行为 0；
- 列起点是 16 的倍数，**或**是一个运行时值——PTO 在运行时下降的正是同一个线性形式，
  两者完全一致，且双方都无法在编译期检查分块对齐。

批量累加器的形状正是围绕这个判定条件设计的。
[`FlattenTileNdTo2D`](14-flatten_tile_nd_to_2d.md#批量累加器按列打包)
把 `tile.batch_matmul_acc` 累加器的 `B` 页打包进一块 `[M, B*N]` 的 Acc tile，第 `b` 页位于
`[0, b*N]` —— 一个列起点块对齐、保留完整行范围的窗口，也就是这里被接受的形状；对于不满足
该条件的 batch 几何（`M % 16 != 0`、`N % 16 != 0`、元素非 4 字节）它会直接报错，而不是发射
一个会被本判定条件静默拒绝的窗口。因此上面的列表不只是"哪些窗口能得到更好的偏移"的描述，
它是那个 pass 编译时所遵循的契约。

一个值得了解的结果：被接受的列窗口在 NZ 布局下是连续的，因此其跨度恰好等于它实际占用的
字节。两个不相交的列窗口因此在生命周期/重叠分析中互不相交；而此前的行主序包络会相互重叠，
导致 MemoryReuse 拒绝在同一循环中携带两个这样的窗口。这也正是同一块打包累加器的 `B` 页能
共存的原因：它们是同一块分配上的 `B` 个互不相交的列窗口。

### 哪些情况保持行主序算术，以及哪些仍是近似

被该判定条件拒绝的所有情况都逐位回退到这些切片一直以来使用的算术。其中两类确实是行主序
稠密的，其余则是已知缺口——在此列出，以免上表被误读为正确性声明：

- **收窄了行范围的 Acc 窗口**（或起始行非 0）。这是已知缺口而非受支持的情形：这类窗口
  **根本不存在**正确的独立基地址，因为它的块列按父张量的 `rows` 跨步，而其自身描述符按
  它自己的 `rows` 跨步。因此该 pass 保留原有的行主序数值，而不是换成另一个同样错误的数值。
  链接在这类窗口之下的切片同理：其祖先本身已不可寻址，任何叶子节点上的局部算术都无法修复。
- **静态列起点落在 16 宽分形块内部的 Acc 窗口。** `CanonicalizeTileSlice` 明确将这类窗口
  列入白名单，视其为合法的 MAD 目标（它完全位于单个块列内部），且在该路径上切片会下降为
  `pto.subview`，此字节偏移是死值——因此这里不能拒绝它。
- **fractal-512 的 Mat / Left / Right 张量。** 它们同样是分块的，**不是**行主序稠密：
  PTO 为其设定 `rowStride = innerCols`、`colStride = rows`，与累加器具有同样形态的分块结构。
  它们的切片偏移存在同样的潜在不匹配，此处刻意不作改动；对其建模属于另一项工作。

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

组 A 和组 B 可能有不同的 MemRef。yield 与 iter_arg 之间的 MemRef 不一致由后续的 MemoryReuse 解决（必要时插入 `tile.move`）。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass InitMemRef();
```

**实现文件**：`src/ir/transforms/init_memref.cpp`

- `NormalizeStmtStructure` 在 MemRef 初始化之前被内部调用
- `MaterializePtoLevel3ScratchMutator` 在所有 tile shape/cast 重写之后、MemRef 收集之前，仅插入缺失的编译器可选 scratch（`tile.ci`、窄化 `tile.cast`、必要的 `tile.sort32`）；`tile.sel` / `tile.sels` / `tile.prelu` 的 caller tmp 原样保留
- `InitMemRefMutator` 从 `TileType` 读取 `memory_space` 并创建 MemRef 对象
  - 处理 view 操作、复用输入操作（`tile.store`、`matmul_acc`、`gemv_acc`）、tile 别名（`a = b`）以及 ForStmt/IfStmt yield 值的 MemRef 共享
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
- 测试按 planner/target 门控的 `tile.ci`、窄化 `tile.cast` 与 `tile.sort32` scratch 物化
- 测试 tile.store 结果与输出参数共享 MemRef
- 测试累加操作（matmul_acc）与累加器输入共享 MemRef
- 测试 ForStmt 循环携带变量的 MemRef 关系（initValue/iter_arg 共享，yield/return_var 共享）
