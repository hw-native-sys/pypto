# CanonicalizeTileSlice Pass

将 `tile.slice` 下沉 (lower) 为规范的 `tile.extract` 形式，使搬运统一走 `pto.textract`——既包括 Mat-resident 切片（折叠进 matmul / `tile.extract` 消费者），也包括那些惰性实例化会破坏源 tile 的 Vec 切片（为 `tile.col_expand_*` 系列实例化，issue #1640、#2010）。

## 概览

结果 tile 位于 `Mem.Mat` 的 `tile.slice` 是一种合法的高层「Mat tile 子窗口」构造。[`FlattenTileNdTo2D`](13-flatten_tile_nd_to_2d.md) 在展开 `tile.batch_matmul` 的 batch 维时，会为每个 batch page 生成一个这样的 slice：page 偏移为 `batch_index * page_rows`；当 batch 前导维为 1 时该偏移为 0、窗口覆盖整个 tile——但它仍然是一个 `tile.slice`。

PTO ISA 支持 Mat 上的 `pto.subview` 作为零拷贝别名（无数据搬运），因此当消费者能直接接受 subview SSA 时，独立的 Mat slice 是合法的。但是，触发惰性实例化（通过 `MaterializeSubviewOperandIfNeeded`）的消费者会尝试生成 `loc=mat → loc=mat` 的 `pto.textract`——这是 Ascend 910C 等目标不支持的 L1→L1 DMA 路径。本 pass 为了效率，通过把可规范化消费者（extract/matmul）对应的 Mat-resident `tile.slice` 偏移折叠进消费者来消除这些 slice，随后删除已死的 slice。消费者不可规范化的 Mat slice（如 `tile.move`）保持原样——它会下沉为合法的 `pto.subview`。

本 pass 还会规范化被 `tile.col_expand_*` 系列消费的 **Vec** `tile.slice`（issue #1640、#2010）。这些算子无法读取 `pto.subview` 操作数，因此 codegen 会通过 `pto.textract` 把该 slice 惰性实例化到 slice 自身的结果缓冲区——而由于 `tile.slice` 继承其源 tile 的内存，该缓冲区**位于仍然存活的源 tile 内部**。于是这次 extract 是在自己的输入上原地执行的，只有当它是一次**恒等拷贝**时才安全。这需要同时满足两个条件：

| 条件 | 何时会不成立 |
| ---- | ------------ |
| 目标**地址**正确 | `AllocateMemoryAddr` 会把 `ConstInt` 偏移折叠成 `base + off`；但**动态**偏移无法编码为 `ConstInt` 地址，会退化到裸源基址——抽出的窗口于是落到源 tile 的第 0 行（#1640）。 |
| 目标**布局**一致 | slice 缓冲区是稠密的（行间距 = slice 列数），而源窗口是跨步的（行间距 = 源列数）。二者只有在窗口**连续**时才相同：单行，或覆盖全部列。多行 tile 的列切片（`t[:, a:b]`）会在自己仍然存活的源上做 跨步 → 稠密 的重排并将其摧毁——只有第 0 行幸存，因为它的稠密目标地址恰好等于其源地址（#2010）。 |

只要任一条件不成立，该操作数就会被替换为新的 `tile.extract(..., target_memory=Vec)`，其结果获得独立、非继承的分配。`tile.extract` 注册为 `not_inplace_safe()`，因此 [`MemoryReuse`](30-memory_reuse.md) 也不会把这块新缓冲区重新放回源 tile 上。若实例化本身就是恒等拷贝，则该 slice 保持原样——它继续共享源缓冲区，而不必付出一份重复分配的代价。

**Pipeline 位置**：紧跟在 [`AutoTileMatmulL0`](14-auto_tile_matmul_l0.md) 之后（此时读取 batch-page slice 的逐迭代 `tile.extract` 已经存在），先于 [`InferTileMemorySpace`](16-infer_tile_memory_space.md)。

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

1. **收集 (Collect)** —— 索引每个 value 为 Mat-resident `tile.slice(src, shape, offset)`（规范 3 参数形式）的 `AssignStmt`。若某 slice 的 `src` 本身又是一个 Mat slice，则进行剥离 (peel) 并累加偏移，使每个条目最终解析为一个非 slice 的 base tile 加上总偏移 `(off_row, off_col)`。带有 `valid_shape` / `drop_dims` 的 slice（4–5 参数）不是普通窗口，跳过。

2. **改写消费者 (Rewrite consumers)** —— 对每个 slice：
   - **`tile.extract(slice, ir, ic, shape)`**（仅 Mat slice） → `tile.extract(base, ir + off_row, ic + off_col, shape)`。extract 直接读取 slice 的源 tile；当两个加数都是 `ConstInt` 时对索引加法做常量折叠。
   - **`tile.matmul` / `tile.matmul_acc` / `tile.matmul_bias` 的操作数**（仅 Mat slice） → 该操作数被替换为一个新的 `tile.extract(base, off_row, off_col, slice_shape, target_memory=Left|Right)`——lhs 操作数用 `Left`，rhs 操作数用 `Right`。（`tile.matmul_acc` 的累加器操作数位于 `Acc`，永远不会是 Mat slice。）
   - **`tile.col_expand_*` 的操作数**（仅 Vec slice） → 当惰性 `pto.textract` 不是恒等拷贝时——即偏移是动态的，或窗口在基 tile 中不连续（行数大于 1 *且* 比基 tile 窄）——该操作数被替换为一个新的 `tile.extract(base, off_row, off_col, slice_shape, target_memory=Vec)`。两个操作数都会检查。常量偏移且窗口连续的 slice 保持原样。

3. **删除死 slice (Drop dead slices)** —— 结果不再被任何使用者引用的 `tile.slice` 被删除。链式 slice（slice 的 slice）只有在消费它的那个 slice 被删除后才会变死，因此该步骤迭代至不动点（迭代次数以 slice 数量为上界）。结束时仍被使用的 slice，说明其消费者不被本 pass 规范化——保持原样，相对 pass 前的 IR 无回退。

本 pass 是 `FunctionPass`；当不存在规范 `tile.slice` 时 function 原样返回。

## 示例

### slice 折叠进 `tile.extract`

[`FlattenTileNdTo2D`](13-flatten_tile_nd_to_2d.md) 为前导维为 1 的 batch 操作数生成的偏移为 0、全形状的 slice：

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

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

**属性**：`include/pypto/ir/transforms/pass_properties.h`（`kCanonicalizeTileSliceProperties`）

**实现**：`src/ir/transforms/canonicalize_tile_slice_pass.cpp`

**Python 绑定**：`python/bindings/modules/passes.cpp`

**测试**：`tests/ut/ir/transforms/test_canonicalize_tile_slice.py`

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
| 链式 Mat `tile.slice`（slice 的 slice） | 剥离；累加偏移 |
| 带 `valid_shape` / `drop_dims` 的 `tile.slice` | 跳过（不是普通窗口）。若这样的 slice 同时不满足上述任一恒等拷贝条件——动态偏移（例如降秩的 `t[i]`）或非连续窗口——并喂给 col-expand op，codegen 会以 `INTERNAL_CHECK` 直接报错，而不是生成会破坏源 tile 的代码 |
| 其他位于 Vec/Left/Right/Acc 的 `tile.slice` | 不处理（无匹配的消费者） |
| 不含规范 `tile.slice` 的 function | 原样返回 |

## 参见

- [`FlattenTileNdTo2D`](13-flatten_tile_nd_to_2d.md) —— 上游 pass；生成本 pass 下沉的 Mat-resident batch-page `tile.slice`
- [`AutoTileMatmulL0`](14-auto_tile_matmul_l0.md) —— 上游 pass；生成消费 batch-page slice 的 `tile.extract`
