# CanonicalizeTileSlice Pass

将 `tile.slice` 下沉 (lower) 为规范的 `tile.extract` 形式，使搬运统一走 `pto.textract`——既包括 Mat-resident 切片（折叠进 matmul / `tile.extract` 消费者），也包括那些惰性实例化会破坏源 tile 的 Vec 切片（为 `tile.col_expand_*` 系列实例化，issue #1640、#2010），以及地址未对齐的 Vec 子视图（issue #1789）。

## 概览

结果 tile 位于 `Mem.Mat` 的 `tile.slice` 是一种合法的高层「Mat tile 子窗口」构造。[`FlattenTileNdTo2D`](14-flatten_tile_nd_to_2d.md) 在展开 `tile.batch_matmul` 的 batch 维时，会为每个 batch page 生成一个这样的 slice：page 偏移为 `batch_index * page_rows`；当 batch 前导维为 1 时该偏移为 0、窗口覆盖整个 tile——但它仍然是一个 `tile.slice`。

PTO ISA 支持 Mat 上的 `pto.subview` 作为零拷贝别名（无数据搬运），因此当消费者能直接接受 subview SSA 时，独立的 Mat slice 是合法的。但是，触发惰性实例化（通过 `MaterializeSubviewOperandIfNeeded`）的消费者会尝试生成 `loc=mat → loc=mat` 的 `pto.textract`——这是 Ascend 910C 等目标不支持的 L1→L1 DMA 路径。本 pass 为了效率，通过把可规范化消费者（extract/matmul）对应的 Mat-resident `tile.slice` 偏移折叠进消费者来消除这些 slice，随后删除已死的 slice。消费者不可规范化的 Mat slice（如 `tile.move`）保持原样——它会下沉为合法的 `pto.subview`。

本 pass 还会规范化被 `tile.col_expand_*` 系列消费的 **Vec** `tile.slice`（issue #1640、#2010）。这些算子无法读取 `pto.subview` 操作数，因此 codegen 会通过 `pto.textract` 把该 slice 惰性实例化到 slice 自身的结果缓冲区——而由于 `tile.slice` 继承其源 tile 的内存，该缓冲区**位于仍然存活的源 tile 内部**。于是这次 extract 是在自己的输入上原地执行的，只有当它是一次**恒等拷贝**时才安全。这需要同时满足两个条件：

| 条件 | 何时会不成立 |
| ---- | ------------ |
| 目标**地址**正确 | `AllocateMemoryAddr` 会把 `ConstInt` 偏移折叠成 `base + off`；但**动态**偏移无法编码为 `ConstInt` 地址，会退化到裸源基址——抽出的窗口于是落到源 tile 的第 0 行（#1640）。 |
| 目标**布局**一致 | slice 缓冲区是稠密的（行间距 = slice 列数），而源窗口是跨步的（行间距 = 源列数）。二者只有在窗口**连续**时才相同：单行，或覆盖全部列。多行 tile 的列切片（`t[:, a:b]`）会在自己仍然存活的源上做 跨步 → 稠密 的重排并将其摧毁——只有第 0 行幸存，因为它的稠密目标地址恰好等于其源地址（#2010）。 |

只要任一条件不成立，该操作数就会被替换为新的 `tile.extract(..., target_memory=Vec)`，其结果获得独立、非继承的分配。`tile.extract` 注册为 `not_inplace_safe()`，因此 [`MemoryReuse`](35-memory_reuse.md) 也不会把这块新缓冲区重新放回源 tile 上。若实例化本身就是恒等拷贝，则该 slice 保持原样——它继续共享源缓冲区，而不必付出一份重复分配的代价。

此外，PTO 向量指令要求 tile 操作数的基址按 32 字节对齐。零拷贝 Vec slice 的起始地址为

```text
base + (off_row * base_cols + off_col) * storage_bits
```

因此 FP32 的列切片 `[:, 1:2]` 只比对齐的源分配多 4 字节。把这个 subview 直接喂给 `tile.muls` 等普通向量算子可能导致设备卡死（#1789）。本 pass 会将地址未对齐的 Vec slice 操作数替换为新的 `tile.extract(..., target_memory=Vec)`；新分配满足对齐要求，而可证明对齐的 slice 仍保持零拷贝。对于动态偏移，若标量 SSA 算术能证明其已知倍数产生的行或列位移满足 32 字节对齐，也会保持零拷贝。

**Pipeline 位置**：紧跟在 [`AutoTileMatmulL0`](17-auto_tile_matmul_l0.md) 之后（此时读取 batch-page slice 的逐迭代 `tile.extract` 已经存在），先于 [`InferTileMemorySpace`](19-infer_tile_memory_space.md)。

**前置属性 (Required)**：`SSAForm`、`SplitIncoreOrch`、`IncoreTileOps`、`TileOps2D`、`NormalizedStmtStructure`。

**产出属性 (Produced)**：与前置属性相同（属性保持不变的改写）。

**失效属性 (Invalidated)**：无。

**何时使用**：一律在默认 tile 阶段流水线中运行。如果不存在规范 `tile.slice`，本 pass 是 no-op。

## API

| C++ | Python | 层级 |
| --- | ------ | ---- |
| `pass::CanonicalizeTileSlice()` | `passes.canonicalize_tile_slice()` | Function 级 |

```python
from pypto.pypto_core import passes

program_canon = passes.canonicalize_tile_slice()(program)
```

## 算法

对每个 InCore 类型的 function，分三个阶段：

1. **收集 (Collect)** —— 索引每个 value 为规范 3 参数形式 `tile.slice(src, shape, offset)` 的 `AssignStmt`。若某 slice 的 `src` 本身又是一个已记录的 slice，则进行剥离 (peel) 并累加偏移，使每个条目最终解析为一个非 slice 的 base tile 加上总偏移 `(off_row, off_col)`。分析前会解析直接的 `ConstInt` SSA 定义及其普通别名，避免字面量偏移在 `ConvertToSSA` 后被误判为动态值；同时保留标量 SSA 定义用于模对齐证明，例如 `block_idx * 32` 虽是动态值，但可静态确定其为 32 个元素的倍数。带有 `valid_shape` / `drop_dims` 的 slice（4–5 参数）不是普通窗口，跳过。

2. **改写消费者 (Rewrite consumers)** —— 对每个 slice：
   - **`tile.extract(slice, ir, ic, shape)`** → `tile.extract(base, ir + off_row, ic + off_col, shape)`。extract 直接读取 slice 的源 tile；当两个加数都是 `ConstInt` 时对索引加法做常量折叠。
   - **`tile.matmul` / `tile.matmul_acc` / `tile.matmul_bias` 的操作数**（仅 Mat slice） → 该操作数被替换为一个新的 `tile.extract(base, off_row, off_col, slice_shape, target_memory=Left|Right)`——lhs 操作数用 `Left`，rhs 操作数用 `Right`。（`tile.matmul_acc` 的累加器操作数位于 `Acc`，永远不会是 Mat slice。）
   - **`tile.col_expand_*` 的操作数**（仅 Vec slice） → 当惰性 `pto.textract` 不是恒等拷贝时——即偏移是动态的，或窗口在基 tile 中不连续（行数大于 1 *且* 比基 tile 窄）——该操作数被替换为一个新的 `tile.extract(base, off_row, off_col, slice_shape, target_memory=Vec)`。两个操作数都会检查。常量偏移且窗口连续的 slice 保持原样。
   - **普通 call 的操作数**（仅 Vec slice，位于 `AssignStmt` 或 `EvalStmt` 中） → 计算 `(base_byte_offset * 8 + (off_row * base_cols + off_col) * storage_bits) mod 256`。取模前会计入已知的具体 MemRef 字节偏移；内存规划哨兵值按对齐的根分配处理，而非常量基址偏移无法在静态证明。分析会沿普通别名、加减法及乘法追踪标量 SSA 定义，以证明动态倍数仍满足对齐。若结果非零或无法证明，则把操作数替换为新的 `tile.extract(base, off_row, off_col, slice_shape, target_memory=Vec)`。`tile.slice` 自身会被跳过，以便剥离链式视图；`tile.extract` 使用上面的直接折叠规则。
   - **SSA 逃逸 (SSA escape)**（仅 Vec slice） → 未对齐 slice 经过普通别名赋值时，在别名定义处进行物化。未对齐的循环初始值在进入循环前物化，并通过其 `IterArg` 替换；由 `yield` 携带的未对齐值则在 yield 前物化。这样别名和循环携带的 SSA 身份无法绕过普通 call 的查找。

3. **删除死 slice (Drop dead slices)** —— 结果不再被任何使用者引用的 `tile.slice` 被删除。链式 slice（slice 的 slice）只有在消费它的那个 slice 被删除后才会变死，因此该步骤迭代至不动点（迭代次数以 slice 数量为上界）。结束时仍被使用的 slice，说明其消费者不被本 pass 规范化——保持原样，相对 pass 前的 IR 无回退。

本 pass 是 `FunctionPass`；当不存在规范 `tile.slice` 时 function 原样返回。

## 示例

### slice 折叠进 `tile.extract`

[`FlattenTileNdTo2D`](14-flatten_tile_nd_to_2d.md) 为前导维为 1 的 batch 操作数生成的偏移为 0、全形状的 slice：

**改写前 (Before)**：

```python
lhs_slice: pl.Tile[[32, 512], pl.INT8, pl.Mem.Mat] = pl.tile.slice(x_mat, [32, 512], [0, 0])
a:         pl.Tile[[32, 256], pl.INT8, pl.Mem.Left] = pl.tile.extract(
    lhs_slice, 0, ko, shape=[32, 256], target_memory=pl.Mem.Left)
```

**改写后 (After)**（slice 被删除；extract 直接读取已加载的 Mat tile）：

```python
a: pl.Tile[[32, 256], pl.INT8, pl.Mem.Left] = pl.tile.extract(
    x_mat, 0, ko, shape=[32, 256], target_memory=pl.Mem.Left)
```

非零的 page 偏移会折叠进 extract 索引——例如偏移为 `[32, 0]` 的 slice 会把 `extract(slice, 0, ko, ...)` 变为 `extract(x_mat, 32, ko, ...)`。

### slice 折叠进 `tile.matmul` 操作数

当 `AutoTileMatmulL0` 未对某个 matmul 做切分（其已是 L0 大小）时，它的 Mat-slice 操作数被直接转换：

**改写前 (Before)**：

```python
lhs_slice: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.slice(lhs_mat, [16, 256], [0, 0])
rhs_slice: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.slice(rhs_mat, [256, 64], [0, 0])
c:         pl.Tile[[16, 64],  pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_slice, rhs_slice)
```

**改写后 (After)**：

```python
lhs_left:  pl.Tile[[16, 256], pl.BF16, pl.Mem.Left]  = pl.tile.extract(
    lhs_mat, 0, 0, shape=[16, 256], target_memory=pl.Mem.Left)
rhs_right: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
    rhs_mat, 0, 0, shape=[256, 64], target_memory=pl.Mem.Right)
c:         pl.Tile[[16, 64],  pl.FP32, pl.Mem.Acc]   = pl.tile.matmul(lhs_left, rhs_right)
```

### Vec slice 实例化进 `tile.col_expand_mul` 操作数（#1640）

本地 tile 的动态偏移 slice 喂给 `col_expand_mul`（`col_expand_add` 同理）：

**改写前 (Before)**：

```python
row:    pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.slice(local, [1, 256], [row_off, 0])
scaled: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(row, gamma_t)
```

**改写后 (After)**（slice 被删除；操作数被实例化到一个全新、非别名的 tile）：

```python
row_ext: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.extract(
    local, row_off, 0, shape=[1, 256], target_memory=pl.Mem.Vec)
scaled:  pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(row_ext, gamma_t)
```

### 多行 Vec tile 的列切片（#2010）

`[16, 128]` tile 上的 `t[:, 64:128]` 是**常量**偏移的 slice，但它的窗口并不连续——16 行，只取源 128 列中的 64 列——因此同样需要实例化：

**改写前 (Before)**：

```python
hi:     pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(t, [16, 64], [0, 64])
scaled: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(hi, gamma_t)
```

**改写后 (After)**：

```python
hi_ext: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(
    t, 0, 64, shape=[16, 64], target_memory=pl.Mem.Vec)
scaled: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(hi_ext, gamma_t)
```

若不做这次改写，`hi` 会在 `t + 256 B`（即 `t` 内部）分配一块稠密的 `[16, 64]` 缓冲区，惰性 `pto.textract` 一边读 `t` 一边把它的跨步列重排进去，从而覆盖掉 `t` 本身。最终只有第 0 行是正确的——这也正是同样的写法在单行 tile 上无害的原因。

### 未对齐的 Vec 列切片（#1789）

FP32 tile 的第 1 列只比对齐的源分配多 4 字节，因此不能直接供 `tile.muls` 使用：

```python
head:   pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.slice(local, [16, 1], [0, 1])
scaled: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.muls(head, 0.5)
```

本 pass 会给向量操作提供一块对齐且独立分配的 tile：

```python
head_ext: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.extract(
    local, 0, 1, shape=[16, 1], target_memory=pl.Mem.Vec)
scaled:   pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.muls(head_ext, 0.5)
```

作为边界情况，FP32 的第 8 列距源基址恰好 32 字节，因此仍保持零拷贝。当源行跨度按 32 字节对齐时，动态行偏移也可保持零拷贝。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

**属性**：`include/pypto/ir/transforms/pass_properties.h`（`kCanonicalizeTileSliceProperties`）

**实现**：`src/ir/transforms/canonicalize_tile_slice_pass.cpp`

**Python 绑定**：`python/bindings/modules/passes.cpp`

**测试**：`tests/ut/ir/transforms/test_canonicalize_tile_slice.py`

## Acc 累加器窗口

本 pass 还会**拒绝**一种它无法修复的形状。L0C 采用 NZ 布局：`[M, N]` tile 的
block `(r_b, c_b)` 位于 `(c_b * M/16 + r_b) * fractal`。因此，只有当窗口覆盖父
tile 的全部行范围，或落在单个 16 列 block 之内时，它才是连续的。跨多个列 block
的 `Acc` tile 的行窗口是跨步的，而 MAD 从裸指针紧凑地写出目的地、没有目的地
stride，于是第一个列 block 之后的每个列 block 都会落到错误的行 tile 上——而且是
静默的，每个行 tile 只有前 16 列是正确的。

与上面的 Vec 情形不同，这里没有可用的修复手段：`Acc` 窗口无法拷出再拷回，因为内存
图中没有任何路径指向 `Acc`。因此，当 matmul 的累加器操作数是非连续的 `Acc` 窗口
时，会抛出 `ValueError`，并在错误信息中给出可用的写法——两者只差在切分的轴上：把
累加器按 `[rows, N * tiles]` 分配，然后切列。

该检查以算子注册表的 `set_output_reuses_input` 为作用域，因此无需逐一列举即可覆盖
`tile.matmul_acc` / `tile.gemv_acc` / `tile.matmul_mx_acc`。对于无法证明的情况它
保持静默：符号化的 extent、非 `Acc` 布局，或无法证明落在单个 block 内的动态列偏移。
无论该算子是绑定到结果（`acc = pl.tile.matmul_acc(...)`）还是写成裸语句
（`pl.tile.matmul_acc(...)`），检查都同样生效。

### 把累加器操作数解析回它所指的窗口

累加器操作数很少就是 `tile.slice` 的结果 Var 本身。把 slice 提到 K 循环之外——这是
最自然的写法，因为窗口不随 K 步变化——会让循环体内的操作数变成一个 `IterArg`，它与
slice 结果是不同的 Var；保持形状的 `tile.reshape` 则是同一段字节的另一个 SSA 名字。
因此本 pass 记录的是**每个 Var 指向哪个窗口**，而不仅仅是"这个 Var 是不是 slice"，
检查时按操作数在该映射中查表。

一个 Var 会沿下列**保持标识**的边继承另一个 Var 的窗口；这些边都在收集 slice 的那
一次程序序遍历中一并记录：

| 边 | 窗口来源 |
| -- | -------- |
| `v = tile.slice(src, shape, offset)` | 窗口本身（种子）；链式 slice 会被剥离并累加偏移 |
| `v = w`（普通 SSA 别名） | `w` 的窗口 |
| `v = <op>(src, ...)`，且该 op 的输出落在某个源操作数的存储上 | 该源操作数的窗口。这一关系来自注册表（`OutputMemoryInheritsInput() && IsInplaceSafe()`，或 `set_output_reuses_input`），而非硬编码的 op 列表——因此 `tile.reshape` / `tile.set_validshape` 以及链式 `tile.matmul_acc` 都被覆盖，而写入**新缓冲区**的 `tile.transpose` 和 `tile.extract` 被排除在外 |
| 循环 `IterArg` | 其**初值**的窗口（`ForStmt` / `WhileStmt`） |
| `ForStmt` / `WhileStmt` 的 `return_vars_[i]` | 循环体末尾 `pl.yield_` 为 `iter_args_[i]` 给出的窗口——**仅当它与初值指向同一窗口时**，即典型的循环携带。零次迭代的循环返回的是初值，因此绑定其他窗口就等于宣称一个某条路径并不指向的窗口 |
| `IfStmt::return_vars_[i]` | 两个分支 yield 的窗口，且可证明两者**相同**（同一父 Var、偏移相等） |

只有当两端**可证明描述同一窗口 extent** 时才记录该边——相同的 rank、相同的 dtype，
且每一维都是取值相同的编译期常量。"输出落在源的存储上"说明两个 Var 指向同一段**字
节**；extent 判定则说明检查从操作数自身 `TileType` 读到的形状仍是这段字节构成的窗口
extent。两者合起来构成证明；任何无法证明的情况都不记录——即保持原有的静默，而不是
猜测：

- **改变形状**的 `tile.reshape` 不满足该判定，因此改形后的 extent 绝不会被拿来与父
  tile 的行数比较；
- `tile.reinterpret_view` 在 dtype 上不满足——16 列 block 的运算假定累加器元素为
  4 字节；
- `tile.transpose_view` 交换末两维 extent，因此对任何非方形窗口都不满足。**方形**窗
  口会通过，且无害：转置视图覆盖完全相同的字节、行列 extent 也完全相同，两条接受规
  则给出的结论与未转置时一致；
- 符号化 extent 不满足该判定，这与检查本身拒绝对符号化形状做推理是一致的。

内存空间刻意不参与比较：结论来自**父** tile 的内存空间与操作数的 extent，从不依赖操
作数自身的内存空间——而这些 tile 视图算子推导出的结果类型本就会把它留空，交给
`InferTileMemorySpace` 后续填写。

#### 循环回边是被检查，而不是被记录

被循环携带的累加器指向两个窗口：第 0 次迭代是初值的窗口，此后每一次迭代都是循环体
末尾 `pl.yield_` 所指的窗口。一个 Var 无法被记录为同时指向两个窗口，而记录其中任何
一个都会让另一次迭代的目的地变得不可见——因此对 yield 的窗口改为按同一规则做**检
查**，目的地的 extent 取自 `IterArg`（循环赋予该携带值的类型）。

三个条件保证它不会误拒：

- 只对循环体真正当作累加器**目的地**使用的携带值生效，因此只被读取的携带值绝不会因
  为一个 MAD 从不写入的形状而被拒绝；
- 只在循环**可证明执行超过一次**时生效——`start` / `stop` / `step` 均为常量且
  `start + step < stop`。只执行一次的循环永远不会走回边；符号化的迭代次数以及所有
  `while` 都算作无法证明，保持静默；
- 它只做检查、从不建立绑定，因此映射保持无环：无需不动点迭代，遍历仍是一次前向扫描。

典型的循环携带（`yield matmul_acc(iter_arg, ...)`）会直接解析回初值的窗口，因此这一
检查得到的结论与循环体内已经通过的结论相同，对流水线生成的形状没有任何额外代价。

**已知边界——全部表现为静默，绝不会产生误拒：**

- **无法证明的迭代次数会让回边不被检查。** 符号化的循环上界或 `while` 无法证明会走
  回边，因此若某个携带值的 yield 指向比初值更糟的窗口，在那里不会被检查到。
- **两个分支指向*不同*窗口的 if 合并不会被绑定。** 绑定其中任何一个都会做出另一条
  路径否定的断言——而该映射同时还服务于链式 slice 的偏移运算，错误的绑定可能改变某
  个无关 slice 的结论。
- **跨函数的路径不在覆盖范围内。** 这是一个 function pass，因此作为 InCore 函数参数
  传入的窗口不携带窗口信息。

### 被拒绝的窗口未必出现在 kernel 源码中

编译器生成的窗口可能在 kernel 源码中根本没有对应的 `tile.slice` 却触发本检查，因此
诊断信息不能假定作者可以去修改它所指的那个 slice。

`FlattenTileNdTo2D` 过去是这类窗口的主要来源：它展开 `tile.batch_matmul_acc` 时把
累加器的各个 batch page **沿行**堆叠，并为每个 batch 切出一页
（`acc_page_i = tile.slice(acc, [M, N], [i * M, 0])`），当 `N > 16` 时这正是被拒绝的
形状。该下降路径已被移除——现在各页**沿列**打包
（`acc_page_i = tile.slice(acc, [M, N], [0, i * N])`），即被接受的"覆盖全部行范围"的
形状；无法按列打包的形状由它自己报错，并给出 DSL 侧的规避方式。参见
[批量累加器按列打包](14-flatten_tile_nd_to_2d.md#批量累加器按列打包)。

窄于等于 16 列的页仍保留旧的行打包形式，而本 pass 把这种窗口作为"单块列"放行，因此
行窗口依然会到达本检查——只是不再是本 pass 会拒绝的那一种。因此该拒绝规则保持原样：
它是防止未来某个 pass 再次引入跨步累加器目的地的兜底，收窄它就会放过一次静默算错。

这是针对上游缺陷（[hw-native-sys/pto-isa#253](https://github.com/hw-native-sys/pto-isa/issues/253)）
的规避，而不是 DSL 的固有属性——`TMATMUL_ACC_IMPL` 把目的地退化为裸 `.data()`
指针，并从左操作数取 `m`，因此 `TileRes::Rows` 从未被读取。若 pto-isa 支持了目的地
stride，应放宽或删除该拒绝，而不是把它保留为长期规则。

## Pass 属性

| 属性 | 取值 |
| ---- | ---- |
| Required | SSAForm、SplitIncoreOrch、IncoreTileOps、TileOps2D、NormalizedStmtStructure |
| Produced | SSAForm、SplitIncoreOrch、IncoreTileOps、TileOps2D、NormalizedStmtStructure |
| Invalidated | — |

## 适用范围

| Op | 处理 |
| -- | ---- |
| 喂给 `tile.extract` 的 Mat-resident `tile.slice`（3 参数） | 折叠进 extract；删除 slice |
| 喂给 matmul 族操作数的 Mat-resident `tile.slice`（3 参数） | 替换为 `tile.extract(target_memory=Left\|Right)`；删除 slice |
| 喂给 `tile.col_expand_*` 的动态偏移 Vec `tile.slice`（3 参数） | 替换为 `tile.extract(target_memory=Vec)`；删除 slice（#1640——地址退化到裸源基址） |
| 喂给 `tile.col_expand_*` 的常量偏移**非连续** Vec `tile.slice`（多行 *且* 比基 tile 窄，如 `t[:, a:b]`） | 替换为 `tile.extract(target_memory=Vec)`；删除 slice（#2010——稠密重排会写在自己仍然存活的源上） |
| 喂给 `tile.col_expand_*` 的常量偏移**连续** Vec `tile.slice`（单行，或覆盖源的全部列） | 保持原样（惰性 textract 是安全的恒等拷贝；继续共享源缓冲区） |
| 喂给普通 call、继承地址无法证明按 32 字节对齐的 Vec `tile.slice` | 替换为 `tile.extract(target_memory=Vec)`；删除 slice（#1789） |
| 喂给普通 call、继承地址可证明按 32 字节对齐的 Vec `tile.slice` | 保持原样；继续使用零拷贝 subview |
| 链式 Mat `tile.slice`（slice 的 slice） | 剥离；累加偏移 |
| 带 `valid_shape` / `drop_dims` 的 `tile.slice` | 跳过（不是普通窗口）。若这样的 slice 同时不满足上述任一恒等拷贝条件——动态偏移（例如降秩的 `t[i]`）或非连续窗口——并喂给 col-expand op，codegen 会以 `INTERNAL_CHECK` 直接报错，而不是生成会破坏源 tile 的代码。上面的 Acc 累加器检查对这类 slice 仍然生效：它只需要物理基址和偏移，而这些信息对每个窗口都会记录，与是否可规范化无关 |
| 用作 matmul **累加器**、且窗口既不覆盖父 tile 全部行范围、也不落在单个 16 列 block 内的 `Acc` `tile.slice` | **拒绝**并抛出 `ValueError`，错误信息给出切列的写法——MAD 没有目的地 stride，该写入会静默落到错误的行 tile 上（pto-isa#253）。操作数不必**就是**该 slice：循环 `IterArg` 携带、普通 SSA 别名，或保持形状的视图 / 原地算子都会被解析回同一窗口（见"把累加器操作数解析回它所指的窗口"） |
| 其他位于 Left/Right/Acc 的 `tile.slice`，含**连续**的 Acc 累加器窗口 | 不处理（无匹配的消费者） |
| 不含规范 `tile.slice` 的 function | 原样返回 |

## 参见

- [`FlattenTileNdTo2D`](14-flatten_tile_nd_to_2d.md) —— 上游 pass；生成本 pass 下沉的 Mat-resident batch-page `tile.slice`
- [`AutoTileMatmulL0`](17-auto_tile_matmul_l0.md) —— 上游 pass；生成消费 batch-page slice 的 `tile.extract`
