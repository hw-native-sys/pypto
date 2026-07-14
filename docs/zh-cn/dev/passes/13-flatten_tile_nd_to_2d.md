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

1. **验证前置条件**：检查静态物理形状、最后轴归约、不允许对 >2D 使用 `tile.read`/`tile.write`/`tile.slice`
2. **变换语句**：遍历函数体，将 >2D Tile 操作转换为 2D，并保留动态的 `valid_shape`（见[动态 valid_shape](#动态-tile-维度issue-1578)）
3. **验证后置条件**：由独立的 `TileOps2D` 属性验证器 (property verifier) 检查改写后的 InCore IR 仅包含受支持的 Tile rank 与 codegen-ready transpose 形态

按语句类型处理：

| Tile 操作 | 变换方式 |
| --------- | -------- |
| `tile.load`（>2D） | 直接将结果类型改为 2D（load 从 rank>2 张量窗口产生 2D tile） |
| `tile.store`（rank>2 张量） | 在转换后 IR 中注入原始张量 rank 对应的分区 `shapes` 作为额外的第 4 个操作数，供后端 codegen 重建 `partition_view`；DSL 源码不变。若 tile 操作数本身仍是 rank>2(例如用户显式 `tile.reshape` 升到 3D 后再喂给 `pl.assemble` 写入 N-D 张量视图),pass 会先插入一个 `tile.reshape` 把 tile 操作数压回 2D —— codegen 要求 tile 必须是 2D,而原始 tile shape 仍由 `shapes` 分区操作数携带 |
| `tile.store`（2D 张量） | 直接透传 |
| `tile.create`/`tile.full`（>2D） | 直接使用展平的 2D 形状重建 |
| `tile.sum`/`tile.max`/`tile.min`（>2D） | 将 axis 映射为 1（2D 的最后轴） |
| `tile.transpose` | `pto.ttrans` scratch 物化的唯一归属。进入时为 3-arg（input, axis1, axis2）。**2D**：创建一块 scratch tile（shape = 源页，位于输入所在 memory），产出 codegen-ready 的 4-arg `tile.transpose(in, a1, a2, scratch)`。**>2D**（末两轴交换）：展开为逐 batch 的 2D transpose，每个都是 4-arg 形态，scratch 从扁平 `[batch*A, B]` 池中切片，再 assemble 进合并后的 2D 输出。交换 batch 轴属用户错误 |
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
        out_0 = pl.store(y_tile, [0, 0, 0], out_0)
        return out_0
```

3D Tile `[2, 3, 4]` 被展平为 `[6, 4]`。`tile.load` 直接产生 2D tile，无需插入 `tile.reshape`。`tile.store` 接受 2D tile 并写入原始的 rank>2 张量。对于 rank>2 张量，Pass 会在转换后 IR 中将原始分区 `shapes` 注入为额外的第 4 个操作数（例如 `pl.store(y_tile, [0, 0, 0], out_0, (2, 3, 4))`）；该操作数仅存在于转换后的 IR 中，不属于 DSL 源码。

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

> chunk 必须放得下片上 Vec（UB）内存（`CHUNK * <保留维> * <存活 tile 字节数> <= UB 容量`），否则
> `AllocateMemoryAddr` 会以 "Vec buffer usage exceeds platform limit" 报错。选 chunk 是用户的责任。

如果一个 >2D tile 到达本 Pass 时**物理形状是动态的**（用户没切静态 chunk），它无法展平,Pass 会抛出可操作的
报错,指向两种修法:用 `pl.range`/`pl.parallel` 对动态维分块,或在进入 InCore（`pl.at`）作用域前 reshape 为 2D。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

实现按职责拆分：

| 阶段 | 文件 | 职责 |
| ---- | ---- | ---- |
| 协调 | `src/ir/transforms/flatten_tile_nd_to_2d/pass.cpp` | 选择 InCore 函数，并按 analysis → rewrite 顺序执行 |
| 分析 (analysis) | `src/ir/transforms/flatten_tile_nd_to_2d/analysis.cpp` | 只读的前置条件验证 |
| 改写协调 | `src/ir/transforms/flatten_tile_nd_to_2d/rewrite.cpp` | 递归遍历语句并分派算子改写 |
| 改写工具 | `src/ir/transforms/flatten_tile_nd_to_2d/rewrite_utils.cpp` | 共享形状、索引和容量辅助逻辑 |
| 批量矩阵乘改写 | `src/ir/transforms/flatten_tile_nd_to_2d/batch_matmul.cpp` | 批量矩阵乘与累加算子的分页降级 |
| 转置改写 | `src/ir/transforms/flatten_tile_nd_to_2d/transpose.cpp` | 独立 N 维转置的降级 |
| 验证 (verification) | `src/ir/transforms/flatten_tile_nd_to_2d/verification.cpp` | 独立验证 `TileOps2D` 后置条件 |

这些阶段入口和改写组件接口仅供 transform 内部使用；公共 API 仍为 `pass::FlattenTileNdTo2D()`。

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
