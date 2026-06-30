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
3. **MemRef 共享**（全局「最大优先 + first-fit」装箱，`IdentifyReuseOpportunities`）：在每个内存空间内，按 **大小从大到小** 装箱；后续每个区间加入第一个其全部成员都能与之共享的缓冲区（生命周期不重叠 + hazard / no-alias 安全，见 `can_share`）。缓冲区的分配大小由其首个（最大）成员固定，因此之后纳入更小的成员是「免费」的 —— 且 *后定义的较大区间* 现在可以承载 *先定义的较小区间*。（此前的定义序贪心带有单向的大小门槛 `source.size >= target.size`，因此两个生命周期不相交、但较小者先定义的 tile 永远无法合并。）每个成员被重定位到的「代表」是该缓冲区的最大成员；由于 InitMemRef 会把所有 `tile.alloc` 提升到函数体头部，代表的 alloc 支配整个函数，因此代表即使定义在其部分成员之后也是安全的。由于装箱器不再按程序序处理，每个成对门槛（hazard、no-alias）都会在两个方向上检查。
4. **循环携带变量重对齐**（`AlignLoopCarriesToInitMutator`）：共享（步骤 3）只会重写由 `AssignStmt` 定义的变量（producer/init），而循环携带的 `iter_arg`/`return_var` 节点被排除在生命周期/共享映射之外、仍保留原始 MemRef。本步骤**自外向内**遍历 `ForStmt`，将每个循环的 `iter_arg`/`return_var` 重对齐到其（已复用的）`initValue` 的 MemRef，并在递归前写入 `var_remap_`，使嵌套循环能观察到已修正的外层 `iter_arg` 作为其 init。若缺少本步骤，被复用的**嵌套流水化 `matmul_acc`** 累加器会分裂到两个 Acc 缓冲区，导致步骤 5 插入非法的 `acc→acc tile.move`，被 Ascend 910B 的 ptoas 拒绝（[#1352](https://github.com/hw-native-sys/pypto/issues/1352)）
5. **Yield 修复**：修复控制流返回变量的 MemRef 不一致：
   - **ForStmt**：确保 4 个循环携带变量（initValue、iter_arg、yield value、return_var）共享同一个 MemRef。若 MemRef 不同则在 yield 前插入 `tile.move`
   - **IfStmt**：修补 return_vars 使其 MemRef 与 yield value 一致
6. **移除冗余 alloc**：收集仍被 TileType 变量引用的所有 MemRef，然后移除不再使用的 `tile.alloc` 语句

**复用条件**：

- 生命周期不重叠（无干涉）。当 `prev.last_use <= curr.def` 时，两个变量不重叠（即源的最后使用可以和目标的定义在同一语句，因为在同一语句内输入先于输出被消费）
- 相同内存空间
- 缓冲区大小取其**最大**成员；由于按最大优先装箱，后纳入的成员都不大于代表，故无需显式字节大小检查（复用方向也不再被限制为「先定义且更大」）
- **No-alias 守护**（算子语义）：定义复用变量的算子可以禁止其输出与某些输入操作数共享缓冲区——因为硬件在**写输出的同时读取**这些输入,原地写会中途破坏该算子。三个来源汇入同一个"每个输出禁止 alias 的输入集合"（`ForbidAliasCollector`）：
  - `not_inplace_safe()` —— 该算子无法以 `src == dst` 运行，因此其输出不得 alias **任何**输入操作数。
  - `forbid_output_alias(i)` —— 该算子对其值操作数 in-place-safe，但在写输出时读取**某个特定**操作数，因此输出不得 alias 该操作数的缓冲区。
  - **升精度 `tile.cast`**（直接在 `ForbidAliasCollector` 处理）—— 输出 dtype 比输入**更宽**时,cast 无法原地：元素 `i` 在 `i*in_bytes` 处读、`i*out_bytes` 处写,写指针超前于读指针,冲掉尚未转换的输入。降精度 / 同宽 cast 仍 in-place-safe（保留下方的跨 dtype 复用）。

  MemoryReuse 拒绝将输出放到任一禁止操作数的**物理缓冲区**上,并通过 reuse-map 合并**与** VIEW 继承（`reshape`/`slice` 共享其源的 MemRef base）解析每个操作数——因此间接到达的禁止操作数（其 owner tile 被复用到别的缓冲区,或经 view 占用）也能被捕获。

  当前声明 no-alias 约束的算子：

  | 算子 | 约束 | 为何输出不能 alias 输入 |
  | ---- | ---- | ----------------------- |
  | `tile.recip`、`tile.rsqrt` | `not_inplace_safe` | 高精度路径在写输出时读取输入**和** tmp scratch |
  | `tile.row_sum` / `row_max` / `row_min` | `not_inplace_safe` | `TROW*` 在写规约输出 `[M, 1]` 时读取整行输入 + tmp scratch |
  | `tile.mrgsort_format1` | `not_inplace_safe` | 归并排序 intrinsic 要求 `src != dst` |
  | `tile.fmod`、`tile.fmods` | `not_inplace_safe` | `TFMOD`/`TFMODS` 按 `a - trunc(a/b)*b` 计算，先用 `dst = a/b` 覆盖输出，再重新读取原始 `src0`（`a`）做最后的减法；当 `dst == src0` 时该减法读到的是已被覆盖的商，导致每个元素都算成 `0` |
  | `tile.transpose` | `not_inplace_safe` | `pto.ttrans` 非 in-place 安全：a2a3 非对齐标量路径直接从 `src` 写 `dst`（不经 tmp 暂存），`dst == src` 会边写边读损坏数据。输出始终分配新 buffer（InitMemRef 也不会为其继承输入的 buffer）。 |
  | `tile.sel` | `forbid_output_alias(0)`（mask）、`(3)`（tmp） | `TSEL` 在写 `dst` 时读取 mask + tmp scratch |
  | `tile.{row,col}_expand{,_mul,_add,_sub,_div}` | `forbid_output_alias(1)`（广播向量） | 行/列向量（arg 1）会被**每个**输出行/列重读,输出若 alias 它则在第一行/列后被覆盖 |
  | `tile.cast`（仅升精度） | 输出 ≠ 输入缓冲区（条件式,在 `ForbidAliasCollector`） | 更宽的输出写指针超前于读指针（见上） |

- **流水线 stage 守卫**（角色感知）：`pl.pipeline(stage=F)` 将循环体复制 `F` 份以实现 ping-pong，`LowerPipelineLoops` 给每个副本产生 tile 的 `Call` 打上 `pipeline_membership` `(group, stage)`（见 [25-lower_pipeline_loops.md](25-lower_pipeline_loops.md)）。`F` 份副本在调度器下并发执行，因此它们程序序不相交的生命周期**不是**安全的复用信号。MemoryReuse 因此在两个 tile 同 group、不同 stage **且至少有一个是 load**（`tile.load` / `tile.read`）时禁止它们共享缓冲——只有 load 缓冲需要 per-stage 私有。不同 stage 的计算中间结果仍可合并,因为禁止*所有*跨 stage 复用（depth = `F`）会超出片上预算（例如 `stage=4` 的 RMSNorm：`4 × 67 KB > 188 KB` UB）。L0 matmul 空间（Left/Right/Acc/Bias）豁免——其缓冲由 `AutoTileMatmulL0` / `CanonicalizeIOOrder` 的 extract 聚类管理,且容量受限。分离是无条件应用的,**绝不为了塞下而静默放松**:若分离后的缓冲超出某空间的片上预算,溢出会以 `AllocateMemoryAddr` 的硬报错暴露出来（kernel 需降低 `stage=` 或 tile 大小）,而不是悄悄回退到合并。复用决策完成后,MemoryReuse 会剥离已消费的 `pipeline_membership` attr,使其不会带到下游 pass 或 codegen。

**不再有 shape / dtype / TileView 兼容性门槛**：共享同一物理 MemRef 的 tile 可以携带**不同**的 shape、dtype 或 `TileView` 属性。PTO codegen 为每个 tile 绑定一条 per-variable 的 `alloc_tile`，因此每个别名都以各自的静态 shape / dtype / layout / `valid_shape` 声明共享基址。这允许例如：

- 跨 dtype 复用 —— BF16 tile 复用已死亡的 FP32 tile 的缓冲区（例如跨 `tile.cast`）；
- `tile.fillpad` 输出复用其输入，以及两个 `pad` 不同的 fillpad 输出共享一个缓冲区；
- N-D tile 在 `valid_shape` 不同的情况下共享缓冲区（各自在自己的 `alloc_tile` 上保留各自的 `valid_shape`）；
- L0 cube 输入 `Left` / `Right` 中 shape 不同的子 tile 共享同一槽位（例如 fused-attention QK 的 `Right` `[k, SEQ]` 被 PV 的 `Right` `[k', HEAD]` 复用，将 L0B 峰值减半 —— issue #1595）。

  早期版本以 `AreTileTypesCompatible`（shape / dtype / view 匹配，外加一个狭窄的 L0 字节复用例外）作为门槛；该门槛已移除。对读-写同体（read-while-write）算子的正确性现由上面的 no-alias 守护精确处理，而不再依赖粗粒度的整块匹配。

**Alloc 清理**：

MemRef 共享完成后，部分 MemRef 对象变为无引用状态（其变量现在指向不同的共享 MemRef）。该 Pass 遍历周围的 `SeqStmts`，移除所有左值 MemRef 指针不在仍使用集合中的 `tile.alloc` `AssignStmt`。

## Ascend910B load + tpop_from_aic 危害

在 `SplitMode` 非 `None` 的 Ascend910B AIV 函数中，如果某个 writer **同时**消费 `tile.load` 的结果（或其合法 view 派生）**和** `tile.tpop_from_aic` 的值，则它的输出不能与该 load 结果落在同一块物理 buffer 上。让 writer 的输出原地复用 load buffer 会在该硬件上产生**静默的错误结果**。

MemoryReuse 掌管所有 buffer 合并决策，因此它从源头上阻止这种危害共享的形成，而不依赖后续的拆分。当 guard 生效时，复用决策在以下条件**同时满足**时被阻止：

- writer 的定义 op 消费了 `tile.tpop_from_aic` 的值，**且**
- 它将要原地复用的那个 buffer 成员（其 last use 正是该 writer 的定义语句）是 load 派生的。

该 guard 由 `BackendHandler::RequiresSplitLoadTpopWorkaround()`（仅 Ascend910B 为 true）以及函数为 split-AIV 这两个条件门控；在其他任何 backend / 函数类型下输入集合为空，复用行为不变。writer 仍可自由复用任何**非** load buffer —— 只有 load + tpop 的原地组合会被拒绝。（该 guard 此前由独立的 `LegalizePTOBufferReuse` pass 在事后拆分 buffer 来实现，现已并入 MemoryReuse。）

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
- 测试字节大小兼容性
- 测试跨 dtype / 跨 `TileView` 复用（现已允许：BF16↔FP32、fillpad 输出↔输入、`valid_shape` 不同）
- 测试 no-alias 守护（`TestForbidOutputAlias` + `TestInplaceOps`），上表每条约束一个用例：
  - `tile.recip` / `tile.rsqrt` / `tile.row_sum` —— 输出不得 alias 输入（`not_inplace_safe`）
  - `tile.sel` —— 输出不得 alias mask / tmp（`forbid_output_alias`）
  - `tile.col_expand_mul` —— 输出不得 alias 广播向量
  - 升精度 `tile.cast` —— 输出不得 alias（更窄的）输入
  - 经 VIEW 间接到达的禁止操作数也被遵守（物理缓冲区解析）
- 测试切片操作的 MemRef 共享保持
- 测试冗余 alloc 语句移除
- 测试控制流生命周期分析（ForStmt 内嵌套 IfStmt、分支变量共享）
