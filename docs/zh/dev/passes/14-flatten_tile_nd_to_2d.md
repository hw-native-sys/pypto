# FlattenTileNdTo2D Pass

将 InCore 函数中的 ND Tile 操作（3D+）展平为 2D，合并除最后一个维度外的所有维度。

## 概述

PTO-ISA 仅支持 2D Tile。`ConvertTensorToTileOps` 之后，Tile 可能具有超过 2 个维度（匹配张量形状）。该 Pass 通过将高维轴合并为一个维度并保持最后一个轴不变，将所有 >2D 的 Tile 操作展平为 2D。例如，Tile `[2, 3, 4]` 变为 `[6, 4]`。

对于 batch 矩阵乘法，`ConvertTensorToTileOps` 会先保留为
`tile.batch_matmul`（带累加器时为 `tile.batch_matmul_acc`）。随后由
`FlattenTileNdTo2D` 统一负责把它展开成带 broadcast 语义的逐 batch
2D `tile.matmul` / `tile.matmul_acc`。

**前置条件**：

- 输入 IR 必须为 SSA 形式
- 输入 IR 必须包含 Tile 操作（需先运行 `ConvertTensorToTileOps`）
- 每个 Tile 的**物理**形状必须为静态（`ConstInt`）；Tile 的 `valid_shape` 可以是动态的，并在展平时
  被保留（见[动态 valid_shape](#动态-tile-维度issue-1578)）
- 所有 Tile 归约操作必须沿最后一个轴归约
- 所有 Tile 内存必须是连续的

**使用时机**：在 `ConvertTensorToTileOps` 之后、`ExpandMixedKernel` / `InitMemRef` 之前运行。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::FlattenTileNdTo2D()` | `passes.flatten_tile_nd_to_2d()` | 函数级 |

**Python 用法**：

```python
from pypto.pypto_core import passes

flatten_pass = passes.flatten_tile_nd_to_2d()
program_2d = flatten_pass(program)
```

## 算法

对每个 InCore 函数（InCore、AIC、AIV）：

1. **验证前置条件**：检查静态物理形状、最后轴归约、不允许对 >2D 使用 `tile.read`/`tile.write`/`tile.slice`，以及不允许写入区域无法连续折叠的 >2D `tile.assemble`
2. **变换语句**：遍历函数体，将 >2D Tile 操作转换为 2D，并保留动态的 `valid_shape`（见[动态 valid_shape](#动态-tile-维度issue-1578)）
3. **验证后置条件**：由独立的 `TileOps2D` 属性验证器 (property verifier) 检查改写后的 InCore IR 仅包含受支持的 Tile rank、2D `tile.assemble` 偏移与 codegen-ready transpose 形态

按语句类型处理：

| Tile 操作 | 变换方式 |
| --------- | -------- |
| `tile.load`（>2D） | 将结果 tile 重建为 2D。对于 natural NZ Mat load，还会在源张量上插入 shape-only 的 2D `tensor.view`，把 leading offsets/shapes/valid_shape 折叠到 2D 源窗口，并要求该窗口按 row-major 连续可折叠。Vec load 和 transposed Mat load 保留原始 rank>2 源窗口，只展平结果 tile |
| `tile.store`（rank>2 张量） | 在转换后 IR 中注入原始张量 rank 对应的分区 `shapes` 作为额外的第 4 个操作数，供后端 codegen 重建 `partition_view`；DSL 源码不变。若 tile 操作数本身仍是 rank>2(例如用户显式 `tile.reshape` 升到 3D 后再喂给 `pl.assemble` 写入 N-D 张量视图),pass 会先插入一个 `tile.reshape` 把 tile 操作数压回 2D —— codegen 要求 tile 必须是 2D,而原始 tile shape 仍由 `shapes` 分区操作数携带 |
| `tile.store`（2D 张量） | 直接透传 |
| `tile.create`/`tile.full`（>2D） | 直接使用展平的 2D 形状重建 |
| `tile.assemble`（>2D 目标） | 用与 `tile.load` 折叠 tensor-rank 偏移相同的行主序折叠，把 ND 偏移折进展平后的 `(row, col)` 空间（`row = ((o0*d1 + o1)*d2 + o2)*… + o[k-2]`，`col = o[k-1]`）；Tile 操作数本身由其定义处的算子展平。要求 source、target 与 offset 具有相同 rank，且写入区域能折叠为连续的行区间（`IsRowMajorCollapseContiguous`），否则在前置条件阶段报错。若不折叠，偏移会以 ND rank 残留在 2D Tile 上，而 codegen 只按位置读取 `elements[0]`/`elements[1]` 并忽略其余元素，从而静默地写到错误地址 |
| `tile.transpose` | `pto.ttrans` scratch 物化的唯一归属。进入时为 3-arg（input, axis1, axis2）。**2D**：创建一块 scratch tile（shape = 源页，位于输入所在 memory），产出 codegen-ready 的 4-arg `tile.transpose(in, a1, a2, scratch)`。**>2D**（末两轴交换）：展开为逐 batch 的 2D transpose，每个都是 4-arg 形态，scratch 从扁平 `[batch*A, B]` 池中切片，再 assemble 进合并后的 2D 输出。交换 batch 轴属用户错误 |
| `tile.batch_matmul` | 展开为逐 batch 的 2D `tile.matmul`，处理 batch broadcast。b_trans/a_trans 操作数以一个零拷贝 `tile.transpose_view`（覆盖在自然 load 之上）出现（不再 transpose-at-load、不搬数据）；tile 级算子本身无 transpose 语义。每个操作数处理方式一致（见下方操作数处理）。**当结果本身就是批量累加器**（下游 `tile.batch_matmul_acc` 会继续写它）时，各页改为通过 `tile.matmul_acc(window, lhs_b, rhs_b, init_cond=True)` 写入同一块按列打包的 `Acc` tile —— 见[批量累加器按列打包](#批量累加器按列打包) |
| `tile.batch_matmul_acc` | 展开为逐 batch 的 2D `tile.matmul_acc`，按 batch 索引取（已展平的）累加器的一个窗口：链按列打包时取 `[M, B*N]` tile 的**列**窗口 `[0, b*N]`，否则取 `[B*M, N]` tile 的旧**行**窗口 `[b*M, 0]` —— 见[批量累加器按列打包](#批量累加器按列打包)。本 pass 未直接确定的内存空间决策（行打包累加器上的 Vec/Acc 来回搬运、上游 `tile.create` 的可重定向生产者改写、TileView 刷新）交由 `InferTileMemorySpace`（pass 20）负责 —— 本 pass 不发射任何 `tile.move` |
| 其他 Tile 操作（>2D） | 替换变量，使用 2D 类型重新创建 |
| 1D/2D Tile 操作 | 不变 |

**统一的操作数处理 —— 整块切片 vs 逐 batch load。** 每个 batch_matmul 操作数
（lhs 或 rhs、转置与否、来自 load 或 move）处理方式完全一致。路由**按操作数**判定：
仅当两个操作数的整块 tile 能一起放进 Mat（L1）（`BatchOperandsWholeFit` 容量门）
**且**该操作数的整块 load 连续可塌（`WholeLoadContiguous`）时才保留整块，否则逐
batch 重发。

- **整块（默认）**：操作数整块进 Mat 一次，再按 batch **切片** —— 普通
  （行批 `[B*rows, cols]`）操作数行切，`tile.transpose_view`（列批 `[K, B*N]`）
  操作数列切。3D `[B, N, K]` 张量的自然 Mat load 在此保留逻辑 ND 源语义，但本
  pass 会在 load 前插入 2D `tensor.view`（`[B*N, K]`），让下游 `tile.load`
  codegen 看到与其他消费者一致的展平源窗口。本 pass 同时把 load 的**结果 tile**
  展平为 2D。广播操作数复用其单页。
- **逐 batch**（整块会撑爆 L1，**或**整块 load 非连续）：从底层自然 `tile.load`
  **逐 batch 重发**（每 batch `[1, .., X, Y]` 窗口 → 2D `[X, Y]`，用 load 自身的
  窗口维度，故部分子 tile 也能正确重发），转置时再加逐 batch
  `tile.transpose_view`。随后丢弃死掉的整块 load/view。
  - *非连续* 指既切多 batch、又部分切矩阵行（中间）维的 load —— 如从 `[2, K, N]`
    切 `[2, K0<K, N]`。展平成 `[2*K, N]` 后各 batch 间有空洞，无法做成单个 2D
    ND2NZ load；逐 batch 后每块是 `[1, K0, N]`（连续），可正常塌。此路由保证
    codegen 的连续性守卫**永不**对 batch_matmul 操作数触发。

**死 load 消除（仅逐 batch）。** 当操作数逐 batch 重发（容量 !fit 或非连续）时，
原始整块 load/view 变为死代码并被丢弃。丢弃 pre-scan 采用与 `LowerBatchMatmul`
**相同的按操作数路由**，故非连续操作数的链在此也被识别为逐 batch。一条链
（`tile.load → tile.transpose_view`，会向上回溯）在其**每一处**使用都是
`tile.batch_matmul[_acc]` 操作数时才可丢弃，且仅当其**所有**消费 matmul 都把它判为
逐 batch 时才丢（与任一保留整块的 matmul 共享的链保持整块）。使用次数按**递归**统计
（含嵌套的 `If`/`For`/`While`/`Scope` 体）。容量门按后端门控（无后端 → 判 fit），
但连续性检查不门控，故非连续路由在单测里也会触发。

> 逐 batch 的 V2C move（move 来源且放不下 L1 的操作数）是后续待办；此类操作数目前
> 仍走整块切片路径，仅在被搬运的整块 tile 放得下固定跨核 ring 时正确。

## 批量累加器按列打包

批量 `tile.batch_matmul_acc` 的累加器是本 pass **唯一**不按通用
`[prod(leading), last]` 规则展平的值。它的各页沿**列**堆叠：一块 `[M, B*N]` 的
`Acc` tile，第 `b` 页位于 `tile.slice(acc, [M, N], [0, b*N])`。

### 为什么不能按行打包

`Acc`（L0C）是 NZ 分块的：对 4 字节累加器元素，`[M, N]` tile 的第 `(r_b, c_b)`
块起始于 `(c_b * M/16 + r_b) * 1024` 字节。因此当 tile 有多个块列时，**行**窗口
是*跨步*的 —— 它的块间距是父 tile 的行数，而不是窗口自身的行数。pto-isa 的 MAD
以裸指针紧凑写出 `[m, n]` 目标，没有目标跨步（hw-native-sys/pto-isa#253），所以按行
打包的 `[B*M, N]` 形状根本没有正确的降级路径：每页只有前 16 列会落在正确位置。
这正是 `CanonicalizeTileSlice`（pass 19）拒绝的形状，参见
[18-canonicalize_tile_slice.md](19-canonicalize_tile_slice.md)。

**列**窗口覆盖父 tile 的整个行范围，因此窗口自身的紧凑几何与父 tile 的几何一致，
被丢弃的跨步也就无关紧要。`GetSliceAccumulatorGeometry` 正是给这种形状计算
NZ 精确字节偏移（参见 [33-init_memref.md](34-init_memref.md)）。

### 生产者侧的变化

`LowerBatchMatmul` 过去把多 batch 结果暂存到 **Vec** —— 在 `Acc` 中汇聚各页需要
L0C→L0C 拷贝，而 ISA 没有这条指令。这会把累加器从 `tile.matmul_acc` 唯一接受的
内存空间里搬走。当结果是某条累加器链的根时，本 pass 现在自己分配打包后的 `Acc`
tile，并按下面的形式写入第 `b` 页：

```text
acc_0 = tile.create([M, B*N], dtype=FP32, target_memory=Acc)
w_b   = tile.slice(acc_b, [M, N], [0, b*N])
m_b   = tile.matmul_acc(w_b, lhs_b, rhs_b, True)   # init_cond=True
acc_b1 = tile.assemble(acc_b, m_b, [0, b*N])       # 自拷贝，codegen 直接省略
```

`init_cond=True` 不需要新算子：字面常量谓词会折叠到**非累加**分支，因此这里发射的
是普通的 `pto.tmatmul ins(lhs, rhs) outs(window)`。这正是 `tile.matmul` 没有目标
操作数、却仍能写入子区域的方式。`tile.assemble` 回写只用于串联 SSA；codegen 识别
出它是同一窗口的自拷贝并且不发射任何指令 —— 这是必需的而不只是优化，因为 `Acc`
目标没有合法的 `tmov`。**因此偏移元组只构造一次，同时传给 slice 和 assemble**：
codegen 通过两个 subview 发射出的 SSA 名字来匹配，重新构造一个数值相同的偏移会
发射出非法的 L0C→L0C 搬运。

对这类结果会关闭 direct-store 融合：把它融合进下一条 `tile.store` 会吞掉链上仍然
需要的语句。

### 消费者（drain）侧的变化

`[M, B*N]` tile 并不是 `[B, M, N]` 输出窗口的行主序折叠，所以单条整块
`tile.store` 会写出错误数据，而下游没有任何环节能发现。打包累加器的 `tile.store`
因此变成**逐页一条 store**，直接从 L0C 写出：

```text
d_b  = tile.slice(acc, [M, N], [0, b*N])
out  = tile.store(d_b, [b, 0, 0], out, [1, M, N])
```

没有 Vec 暂存，也没有 `tile.move`：源为 `Acc` tile 的 store 被判定为 CUBE，因此留在
cube 通道上，不涉及 `ExpandMixedKernel` 的 AIC→AIV 边界。（该边界 —— `Acc`→Vec
搬运 —— 在所有*非*累加器的 `tile.batch_matmul` 上保持不变，它们仍经 Vec 暂存。）

### 何时打包一条链

这个决策在任何改写之前、按整个函数做一次（`acc_packing.cpp`），因为一条链通常跨
多个块：`tile.create` 在 K 循环外，`tile.batch_matmul_acc` 在循环内，`tile.store`
在循环后。改写主循环自带的预扫描是按块进行的，看不到这种结构。

一条*链*是同一缓冲区别名图的连通分量 —— 包括 `tile.batch_matmul_acc` 的原地边、
普通 SSA 别名、循环 `iter_arg` / `yield` / `return_var` 携带边，以及 `IfStmt` 汇合边。
只有当**每个**成员都由本 pass 能逐页改写的形式产生和消费（`tile.create` 或
`tile.batch_matmul` 根、`tile.batch_matmul_acc`、携带边、`tile.store` drain），
且几何形状是 L0C 能寻址的，才会打包：

| 条件 | 原因 |
| ---- | ---- |
| `M % 16 == 0` 且 `N % 16 == 0` | 父 tile 两个方向都必须是完整的 16×16 块，且页的列原点 `b*N` 必须块对齐，否则 `GetSliceAccumulatorGeometry` 会拒绝，`InitMemRef` 静默退回行主序算法 |
| 4 字节累加器元素（FP32 / INT32） | 只有每元素 4 字节时，16×16 块才正好是 `kAccFractal`（1024）字节 |
| `B*M*N*4` 能放进 `Acc` | 从后端读取（`GetMemSize(Acc)`），不写死 —— L0C 在 Ascend910B 上是 128 KB，在 950 上是 256 KB |
| 无部分 `valid_shape` | 第 `b` 页的有效区域从 `b*N` 开始但只有 `N_valid` 宽，打包后的父 tile 没有单一有效矩形可以表达 |
| `batch_count > 1` | `B == 1` 时两种打包都是同一块 `[M, N]` tile，因此现有快速路径逐字节不变 |

不满足其中任一条的链保留旧的行打包降级 —— 当每页不超过 16 列时它仍然正确，因为
此时窗口落在单个 L0C 块列内，而 `CanonicalizeTileSlice` 明确放行这种形状。注意
`M % 16 == 0` 比行打包所需的 `B*M % 16 == 0` **更严格**，所以例如 `M = 8, B = 2,
N = 16` 会有意退回行打包。

另有三种情况连行打包也无法表达。前两种无论页宽多少（包括 16）都会在这里直接报错；
第三种只在超过 16 列时出现，此时行打包本就不可用：

| 情况 | 为什么行打包也救不了 |
| ---- | -------------------- |
| 链中存在多于一个分配型定义（例如 `if k == 0: acc = matmul(...) else: acc = matmul_acc(acc, ...)`，或循环体内重新创建累加器） | 两个缓冲区在控制流汇合点相遇，合并它们需要 ISA 并不具备的 L0C→L0C 拷贝。若放行，程序会一直走到二十个 pass 之后的 `MemoryReuse` `YieldFixup` 并以*内部错误*崩溃 |
| 定义方根本无法写 `Acc`（例如 `tile.load`） | 与 batch 无关 —— 同样的累加器在 `B == 1` 时以完全相同的方式失败 |
| `N > 16` 时以 `tile.move` 排空 | 把 move 按页拆开很容易，难的是把页**聚合**回去：搬出的页保留 L0C 的 `col_major`/1024 块布局，而 `tile.assemble` 无法把它写进 `[B*M, N]` 展平所期望的 `row_major` 向量 tile。这**不是**累加器特有的限制 —— 普通的 `batch > 1` `tile.batch_matmul` 后接任意向量算子会在同一处检查失败（`pto_ops_shared.cpp`，"blayout mismatch between source and result"），所以这里不做任何打包 |

这三种情况各有自己的诊断信息和规避方式；对它们**不会**打印按列打包的说明，因为
它们都与页的几何无关。

其余情况 —— 更宽但无法按列打包的页，或生产者不得不经 Vec 暂存的页 —— 会在**这里
直接报错**，并给出 DSL 侧的规避方式，而不是发射一个地址会静默退回行主序的窗口：

```text
tile.batch_matmul_acc: cannot lower a batch-2 accumulator of 16x24 FP32 pages,
because the page column extent N=24 is not a multiple of 16.
The pages of a batched accumulator have to be packed along COLUMNS — one 16x48
Acc (L0C) tile with page b at tile.slice(acc, [16, 24], [0, b * 24]) — because
the hardware MAD writes its destination compactly and has no destination stride
...
Workarounds: write the batch loop out in the kernel and accumulate each page
into its own 2-D tile (pl.matmul / pl.matmul_acc on 2-D operands); or keep the
accumulator at most 16 columns wide, which fits a single L0C block column and
needs no packing.
```

## 示例

**之前**：

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def main_incore_0(self, x: pl.Tensor[[2, 3, 4], pl.FP32],
                      out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
        x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
        y_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
        out_0 = pl.store(y_tile, [0, 0, 0], out_0)
        return out_0
```

**之后**：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def main_incore_0(self, x: pl.Tensor[[2, 3, 4], pl.FP32],
                      out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
        x_tile: pl.Tile[[6, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
        y_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
        out_0 = pl.store(y_tile, [0, 0, 0], out_0)
        return out_0
```

3D Tile `[2, 3, 4]` 被展平为 `[6, 4]`。`tile.load` 直接产生 2D tile，无需插入 `tile.reshape`。`tile.store` 接受 2D tile 并写入原始的 rank>2 张量。对于 rank>2 张量，Pass 会在转换后 IR 中将原始分区 `shapes` 注入为额外的第 4 个操作数（例如 `pl.store(y_tile, [0, 0, 0], out_0, (2, 3, 4))`）；该操作数仅存在于转换后的 IR 中，不属于 DSL 源码。

## 动态 Tile 维度（issue #1578）

硬件 Tile 对应固定大小的片上缓冲，因此每个**物理** Tile 维度都必须是编译期常量；运行时实际范围保存在
`TileView.valid_shape` 中。要处理动态维，用户**自己写分块循环**：用 `pl.range` 以静态 `CHUNK` 步进迭代
动态维，每趟把这一块 load 成静态物理 `[1, CHUNK, 512]` 的 tile，并在 `valid_shape` 里用
`min(CHUNK, s - c)` 夹住尾块。chunk 大小由用户决定 —— 它对性能影响显著，因此 Pass 不自动选取：

```python
# 用户自己写：对动态 S 维分块，在 valid_shape 里夹住尾块。
for c, (o,) in pl.range(0, s_dim, CHUNK, init_values=(out,)):
    valid = pl.min(CHUNK, s_dim - c)
    t = pl.load(x, [b, c, 0], [1, CHUNK, 512], valid_shape=[1, valid, 512])
    t = pl.cast(t, target_type=pl.FP32)
    o = pl.store(t, [b, c, 0], o)        # 物理静态 [1, CHUNK, 512]，valid 动态
    pl.yield_(o)
```

每趟的 tile 物理上是 `[1, CHUNK, 512]`（静态），`valid_shape` 是 `[1, min(CHUNK, s - c), 512]`（动态）。
**FlattenTileNdTo2D 在这里的唯一职责,就是把这个 >2D tile 降成 `[CHUNK, 512]`,同时保留动态的
`valid_shape`** —— `ComputeMergedValidShape` 用与 `ComputeMergedShape` 合并物理形状相同的方式合并
`valid_shape` 的前导维,但允许动态项,因此运行时尾块能穿过展平活下来,而不是被重置成满物理形状。循环是
用户写的,Pass **不**生成它。

> chunk 必须放得下片上 Vec（UB）内存（`CHUNK * <保留维> * <存活 tile 字节数> <= UB 容量`），否则
> `AllocateMemoryAddr` 会以 "Vec buffer usage exceeds platform limit" 报错。选 chunk 是用户的责任。

如果一个 >2D tile 到达本 Pass 时**物理形状是动态的**（用户没切静态 chunk），它无法展平,Pass 会抛出可操作的
报错,指向两种修法:用 `pl.range`/`pl.parallel` 对动态维分块,或在进入 InCore（`pl.at`）作用域前 reshape 为 2D。

## 循环携带值的 valid_shape 修复

当 `tile.batch_matmul` 的左操作数带着被收窄的 `valid_shape` 时，展开后得到的 2D matmul
结果比它流入的累加器更窄。而累加器所经过的循环携带值**只按其初值定型**，因此它仍然宣称
种子那个没有任何 `mad` 写满过的完整盒高：

```text
acc__tile      : Tile[[64, 256], INT32]                              <- pl.create_tensor 种子
  iter_arg     : Tile[[64, 256], INT32]                              <- 按种子定型
  yield        : Tile[[64, 256], INT32, Acc, valid=[v, 256], compact] <- 循环体实际产出
  return_var   : Tile[[64, 256], INT32]                              <- 被强行拉回 iter_arg 的类型
```

`mad` 以 `ceil(v/16)*16` 的 N-fractal 步长把乘积写进 L0C，而相信完整盒高的读者会按物理行
步长遍历该缓冲区，从而打乱第一个之后的每一个 N-fractal（issue #2470）。因此本 pass 在返回
之前会对每个改写过的函数调用 `narrow_loop_carry::NarrowAccCarries`：把种子按 yield 可证明
的范围重新声明——`tile.create(compact=True)` 加 `tile.set_validshape`，与 `AutoTileMatmulL0`
切分 K 时构造的形式一致——并让循环体的 def-use 闭包通过算子自身的 deducer 重新定型。

在这里修复而不是留给后续 pass，正是流水线可验证性的前提：否则本 pass 产出的携带值会被
`TypeCheck` 诊断与 `AccCompactValid` 属性验证器拒绝。`ConvertTensorToTileOps` 出于同样的
原因调用同一个 helper——2D 种子在 `tensor.matmul` 变成 `tile.matmul` 时就已经被收窄了。

两种情况下携带值保持原样：一是缓冲区的两种读法本来就不会分歧——单 fractal 块的 `[16, N]`
累加器无论有效行是多少都按物理行打包；二是收窄用的表达式只在循环体内计算，重新声明的种子
在那之前根本命名不到它。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

实现按职责拆分：

| 阶段 | 文件 | 职责 |
| ---- | ---- | ---- |
| 协调 | `src/ir/transforms/flatten_tile_nd_to_2d/pass.cpp` | 选择 InCore 函数，按 analysis → rewrite 顺序执行，并修复被改写收窄的循环携带值 |
| 分析 (analysis) | `src/ir/transforms/flatten_tile_nd_to_2d/analysis.cpp` | 只读的前置条件验证 |
| 改写协调 | `src/ir/transforms/flatten_tile_nd_to_2d/rewrite.cpp` | 递归遍历语句并分派算子改写 |
| 改写工具 | `src/ir/transforms/flatten_tile_nd_to_2d/rewrite_utils.cpp` | 共享形状、索引和容量辅助逻辑 |
| 累加器打包 | `src/ir/transforms/flatten_tile_nd_to_2d/acc_packing.cpp` | 按整个函数决定哪些批量 `Acc` 累加器链按列打包 |
| 批量矩阵乘改写 | `src/ir/transforms/flatten_tile_nd_to_2d/batch_matmul.cpp` | 批量矩阵乘与累加算子的分页降级 |
| 转置改写 | `src/ir/transforms/flatten_tile_nd_to_2d/transpose.cpp` | 独立 N 维转置的降级 |
| 验证 (verification) | `src/ir/transforms/flatten_tile_nd_to_2d/verification.cpp` | 独立验证 `TileOps2D` 后置条件 |

这些阶段入口和改写组件接口仅供 transform 内部使用；公共 API 仍为 `pass::FlattenTileNdTo2D()`。

**Python 绑定**：`python/bindings/modules/passes.cpp`

**测试**：`tests/ut/ir/transforms/test_flatten_tile_nd_to_2d.py`、`tests/ut/ir/transforms/test_narrow_loop_carry_valid_shape.py`（携带值修复）、`tests/st/codegen/dsl/test_flatten_dynamic_tile_3d.py`（issue #1578 端到端）

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| 所需 | SSAForm, IncoreTileOps, NormalizedStmtStructure |
| 产生 | SSAForm, TileOps2D, NormalizedStmtStructure |
| 失效 | — |

## 作用范围

| Tile 维度 | 处理方式 |
| --------- | -------- |
| 1D | 不变 |
| 2D | 不变 |
| 3D+ | 展平为 2D |

仅处理 InCore 类型函数（InCore、AIC、AIV）。Orchestration 和 Opaque 函数原样返回。
