# FlattenTileNdTo2D Pass

将 InCore 函数中的 ND Tile 操作（3D+）展平为 2D，合并除最后一个维度外的所有维度。

## 概述

PTO-ISA 的 tile buffer 严格为 2D。`ConvertTensorToTileOps` 之后，逻辑 Tile 可能具有超过 2 个维度（匹配张量形状）。该 Pass 通过将高维轴合并为一个维度并保持最后一个轴不变，将所有 >2D 的 Tile 操作展平为 2D。例如，Tile `[2, 3, 4]` 变为 `[6, 4]`。逻辑 rank-1 tile 在 IR 中保持 rank-1，由 codegen 规范化为物理 `[1, N]` buffer。

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
- N-D `tile.transpose_view` 必须只作为 batch-matmul 操作数被消费；独立的零拷贝
  view 无法表示所需的展平后置换

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

1. **验证前置条件**：检查静态物理形状、最后轴归约、不允许对 >2D 使用 `tile.read`/`tile.write`/`tile.slice`
2. **变换语句**：遍历函数体，将 >2D Tile 操作转换为 2D，并保留动态的 `valid_shape`（见[动态 valid_shape](#动态-tile-维度issue-1578)）

按语句类型处理：

| Tile 操作 | 变换方式 |
| --------- | -------- |
| `tile.load`（>2D） | 直接将结果类型改为 2D（load 从 rank>2 张量窗口产生 2D tile） |
| `tile.store`（rank>2 张量） | 在转换后 IR 中注入与原始张量同 rank 的**有效分区**作为额外第 4 个操作数，供后端 codegen 重建 `partition_view`；它不是原始物理 `shapes`。该分区保留展平前 tile 的有效 extent，仅当原始 tile rank 小于张量 rank 时才在前面补单位轴。DSL 源码不变。若 tile 操作数仍为 rank>2（例如用户将 `tile.reshape` 到 3D 后写入 N-D tensor view），则先插入 `tile.reshape` 将其展平为 2D |
| `tile.store`（2D 张量） | 直接透传 |
| `tile.create`/`tile.full`（>2D） | 直接使用展平的 2D 形状重建 |
| `tile.sum`/`tile.max`/`tile.min`（>2D） | 将 axis 映射为 1（2D 的最后轴） |
| `tile.transpose` | `pto.ttrans` scratch 物化的唯一归属。进入时为 3-arg（input, axis1, axis2）。**2D**：创建一块 scratch tile（shape = 源页，位于输入所在 memory），产出 codegen-ready 的 4-arg `tile.transpose(in, a1, a2, scratch)`。**>2D**（末两轴交换）：展开为逐 batch 的 2D transpose，每个都是 4-arg 形态，scratch 从扁平 `[batch*A, B]` 池中切片，再 assemble 进合并后的 2D 输出。交换 batch 轴属用户错误 |
| `tile.transpose_view` | 2D view 保持不变。N-D view 只允许出现在完全由 `tile.batch_matmul[_acc]` 消费的链上，因为其逐页提取器理解列批表示；其他独立 N-D 用法会被拒绝，单个 2D 零拷贝 transpose 会改变逻辑置换 |
| `tile.batch_matmul` | 展开为逐 batch 的 2D `tile.matmul`，处理 batch broadcast。b_trans/a_trans 操作数以一个零拷贝 `tile.transpose_view`（覆盖在自然 load 之上）出现（不再 transpose-at-load、不搬数据）；tile 级算子本身无 transpose 语义。每个操作数处理方式一致（见下方操作数处理） |
| `tile.batch_matmul_acc` | 展开为逐 batch 的 2D `tile.matmul_acc`，按 batch 索引切分（已展平的）累加器。累加器上的内存空间决策（Vec/Acc 来回搬运、上游 `tile.create` 的可重定向生产者改写、TileView 刷新）交由 `InferTileMemorySpace`（pass 17）负责 —— 本 pass 不再发射任何 `tile.move` |
| 其他 Tile 操作（>2D） | 替换变量，使用 2D 类型重新创建 |
| 1D/2D Tile 操作 | 不变 |

**统一的操作数处理 —— 整块切片 vs 逐 batch load。** 每个 batch_matmul 操作数
（lhs 或 rhs、转置与否、来自 load 或 move）处理方式完全一致。路由**按操作数**判定：
仅当两个操作数的整块 tile 能一起放进 Mat（L1）（`BatchOperandsWholeFit` 容量门）
**且**该操作数的整块 load 连续可塌（`WholeLoadContiguous`）时才保留整块，否则逐
batch 重发。

- **整块（默认）**：操作数整块进 Mat 一次，再按 batch **切片** —— 普通
  （行批 `[B*rows, cols]`）操作数行切，`tile.transpose_view`（列批 `[K, B*N]`）
  操作数列切。3D `[B, N, K]` 张量的自然 Mat load 在此**保留 ND 源窗口**；硬件
  ND2NZ「2 维 GlobalTensor」塌成 `[B*N, K]` 的处理由 `tile.load` codegen 负责
  —— 当 load 结果为 NZ Mat tile 时触发，并在那里发射 2D `make_tensor_view`，故本
  pass 只把 load 的**结果 tile** 展平为 2D。广播操作数复用其单页。
- **逐 batch**（整块会撑爆 L1，**或**整块 load 非连续）：从底层自然 `tile.load`
  **逐 batch 重发**（每 batch `[1, .., X, Y]` 窗口 → 2D `[X, Y]`，用 load 自身的
  窗口维度，故部分子 tile 也能正确重发），转置时再加逐 batch
  `tile.transpose_view`。随后丢弃死掉的整块 load/view。
  - *非连续* 指既切多 batch、又部分切矩阵行（中间）维的 load —— 如从 `[2, K, N]`
    切 `[2, K0<K, N]`。展平成 `[2*K, N]` 后各 batch 间有空洞，无法做成单个 2D
    ND2NZ load；逐 batch 后每块是 `[1, K0, N]`（连续），可正常塌。此路由保证
    codegen 的连续性守卫**永不**对 batch_matmul 操作数触发。

有效性在 staging 边界仍是权威信息：batch-matmul 与 transpose 的 staging pool
携带展平后的有效 extent，逐 batch reload / slice 则从源窗口推导每页有效性。
因此动态尾块会继续保留，而不会被重置成完整 staging tile。

普通 batch-matmul 结果仍必须能表示为单个连续的展平矩形。唯一的有意例外是：
当紧随其后的语句直接 store 该结果时，direct-store fusion 会为每个 batch 页发射
一个独立的原 rank 分区。因此逻辑 `[B, m<M, N]` 可安全写成 `B` 个 `[1, m, N]`
分区，尽管把它作为单个 `[B*M, N]` tile 会形成跨步区域而被拒绝。每个分区使用
该页的有效 M/N extent，而非物理容量。

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
        out_0 = pl.store(y_tile, [0, 0, 0], out_0, (2, 3, 4))
        return out_0
```

3D Tile `[2, 3, 4]` 被展平为 `[6, 4]`。`tile.load` 直接产生 2D tile，无需插入 `tile.reshape`；`tile.store` 接受 2D tile 并写入原始 rank>2 张量。对于 rank>2 张量，Pass 会注入同 rank 的有效分区作为第 4 个操作数。此处展平前 tile 的有效 extent 为 `[2, 3, 4]`，因此分区保持 `(2, 3, 4)`；只有原始 tile rank 小于张量 rank 时才会在前面补单位轴。该操作数只属于转换后的 IR，不是 DSL 源码；这里它恰好等于物理 shape，是因为原始 tile 完全有效。

## 动态 Tile 维度（issue #1578）

硬件 Tile 对应固定大小的片上缓冲，因此每个**物理** Tile 维度都必须是编译期常量；运行时实际范围保存在
`TileView.valid_shape` 中。要处理动态维，用户**自己写分块循环**：用 `pl.range` 以静态 `CHUNK` 步进迭代
动态维，每趟把这一块 load 成静态物理 `[1, CHUNK, 512]` 的 tile，并在 `valid_shapes` 里用
`min(CHUNK, s - c)` 夹住尾块。chunk 大小由用户决定 —— 它对性能影响显著，因此 Pass 不自动选取：

```python
# 用户自己写：对动态 S 维分块，在 valid_shapes 里夹住尾块。
for c, (o,) in pl.range(0, s_dim, CHUNK, init_values=(out,)):
    valid = pl.min(CHUNK, s_dim - c)
    t = pl.load(x, [b, c, 0], [1, CHUNK, 512], valid_shapes=[1, valid, 512])
    t = pl.cast(t, target_type=pl.FP32)
    o = pl.store(t, [b, c, 0], o)        # 物理静态 [1, CHUNK, 512]，valid 动态
    pl.yield_(o)
```

每趟的 tile 物理上是 `[1, CHUNK, 512]`（静态），`valid_shape` 是 `[1, min(CHUNK, s - c), 512]`（动态）。
**FlattenTileNdTo2D 在这里的唯一职责,就是把这个 >2D tile 降成 `[CHUNK, 512]`,同时保留动态的
`valid_shape`** —— `ComputeMergedValidShape` 用与 `ComputeMergedShape` 合并物理形状相同的方式合并
`valid_shape` 的前导维,但允许动态项,因此运行时尾块能穿过展平活下来,而不是被重置成满物理形状。循环是
用户写的,Pass **不**生成它。

**合法性前置条件。** 只有当 ND 有效区映射为展平后行的一个*连续*前缀（即 `(valid_row, valid_col)` 唯一能
表达的形态）时,合并才成立。把行维（除最内维外的所有维）按从高位到低位读取,该条件成立当且仅当存在唯一一个
部分有效的"自由"行维：它之前的每个行维要么由 `valid == 1` 钉住，要么由
`shape == 1` 钉住；它之后的每个行维都完全有效（`valid == shape`）。物理单位轴
即使有效 extent 是符号表达式也安全：良构边界把它限制在 0 或 1，因此它只控制
空/非空。上面的 strip-mine 情形满足该规则——前导物理 `1` 钉住下标，中间的
`CHUNK` 维是自由维。若某个*中间*维在非单位外层维之下仍部分有效（例如物理
`[2, 4, 8]` 配 valid `[2, 2, 8]`），则会得到跨步区域并以 `ValueError` 拒绝；
否则乘积折叠会把 batch-0 的填充标为有效、把真实的 batch-1 数据标为无效。

**空区域豁免。** *空*的有效区 —— 任意维可证为 `0`,例如 `pl.create_tile(valid_shape=[0, 0, 0])` 累加器
（D2 的"尚无有效数据"）—— 表示空集，它平凡地是一个长度为 0 的连续前缀。否则，
钉住检查会把可证为空的维误判为部分有效的中间维并给出误导性报错，因此任意维为 0 时跳过该检查。
乘积折叠随后自行正确处理空区域:某个行维为 0 折叠成 `valid_row = 0`,某个列维为 0 折叠成 `valid_col = 0` ——
无论哪种,都是零个有效元素。

> chunk 必须放得下片上 Vec（UB）内存（`CHUNK * <保留维> * <存活 tile 字节数> <= UB 容量`），否则
> `AllocateMemoryAddr` 会以 "Vec buffer usage exceeds platform limit" 报错。选 chunk 是用户的责任。

如果一个 >2D tile 到达本 Pass 时**物理形状是动态的**（用户没切静态 chunk），它无法展平,Pass 会抛出可操作的
报错,指向两种修法:用 `pl.range`/`pl.parallel` 对动态维分块,或在进入 InCore（`pl.at`）作用域前 reshape 为 2D。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

**实现文件**：`src/ir/transforms/flatten_tile_nd_to_2d_pass.cpp`

**Python 绑定**：`python/bindings/modules/passes.cpp`

**测试**：`tests/ut/ir/transforms/test_flatten_tile_nd_to_2d.py`、`tests/st/codegen/dsl/test_flatten_dynamic_tile_3d.py`（issue #1578 端到端）

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| 所需 | SSAForm, IncoreTileOps |
| 产生 | SSAForm, TileOps2D |
| 失效 | — |

## 作用范围

| Tile 维度 | 处理方式 |
| --------- | -------- |
| 1D | 不变 |
| 2D | 不变 |
| 3D+ | 展平为 2D |

仅处理 InCore 类型函数（InCore、AIC、AIV）。Orchestration 和 Opaque 函数原样返回。
