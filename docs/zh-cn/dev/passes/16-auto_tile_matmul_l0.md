# AutoTileMatmulL0 Pass

针对右操作数为 Mat（左操作数为 Mat 或 Vec）的 `tile.matmul` / `tile.matmul_acc` 进行 L0 切分：从当前 backend 的 L0 容量中挑选 L0 tile 形状 `(m, n, k)`，并把这次 matmul 调用改写成一个 2 阶段流水化的 K-loop，每个迭代用 `tile.extract` 从 Mat 抽取 Left/Right 操作数。当 `[M, N]` 输出本身超过 L0c 时，再对输出做切分（M/N 切分），拆成 `[m, n]` 子块的网格，每个子块直接 store 到输出张量。

## 概览

由 `ConvertTensorToTileOps` + [`FlattenTileNdTo2D`](15-flatten_tile_nd_to_2d.md) 生成的 Mat-resident matmul 通常带有完整的 `(M, N, K)` 操作数形状——几乎一定大于 cube unit 的 L0a/L0b/L0c 容量。本 pass 选取一个能放进 L0 的 `(m, n, k)`，并把该 matmul 改写成一个 K-loop：循环体内用 `tile.extract` 把 `[m, k]` 与 `[k, n]` 的切片送入 `Left` / `Right`，并把累加器写入 `Acc`-resident 的 iter-arg。该循环带有 `ForKind::Pipeline` 与 `pipeline_stages=2`，使下游 [`LowerPipelineLoops`](28-lower_pipeline_loops.md) 可对每次迭代的操作数 `tile.extract` 生成 2 级 ping-pong。

**K 切分 vs M/N 切分。** 当 chooser 返回 `m == M` 且 `n == N` 时，输出已能放进 L0c，因此只切分 K 维（一个 K-loop）。当返回 `m < M` 或 `n < N` 时，`[M, N]` 输出 Acc 会超过 L0c。由于操作数已经是 Mat-resident，*只有*输出溢出：本 pass 把**输出**切成 `ceil(M/m) × ceil(N/n)` 的 `[m, n]` 子块网格（边界处为部分块——`m`/`n` 不必整除 `M`/`N`），每个子块用同样的流水化 K-loop 计算，并把每个 `[m, n]` 的 Acc 子块直接 store 到 `out[mi:, ni:]`（direct-store / 输出落 DDR 的路径）。这样每个 Acc tile 都 ≤ L0c，matmul 能顺利通过 `AllocateMemoryAddr` 而不溢出。输出张量以 SSA 形式在各子块 store 间串联（`out → out_t0 → out_t1 → …`）。

**Pipeline 位置**：紧跟在 [`FlattenTileNdTo2D`](15-flatten_tile_nd_to_2d.md) 之后，先于 [`InferTileMemorySpace`](18-infer_tile_memory_space.md)。此时 tile op 已是 2D，但 memory space 尚未推断。

**前置属性 (Required)**：`SSAForm`、`SplitIncoreOrch`、`IncoreTileOps`、`TileOps2D`、`NormalizedStmtStructure`。

**产出属性 (Produced)**：与前置属性相同（属性保持不变的改写）。

**失效属性 (Invalidated)**：无。

**何时使用**：一律在默认 tile 阶段流水线中运行。如果不存在超过 backend L0 容量的 Mat-resident matmul，本 pass 是 no-op。

## API

| C++ | Python | 层级 |
| --- | ------ | ---- |
| `pass::AutoTileMatmulL0()` | `passes.auto_tile_matmul_l0()` | Program 级 |

```python
from pypto.pypto_core import passes

l0_tile_pass = passes.auto_tile_matmul_l0()
program_tiled = l0_tile_pass(program)
```

## 算法

对每个 InCore 函数中的 `tile.matmul` 或 `tile.matmul_acc`：

1. **过滤** —— 操作数布局：`tile.matmul` 为 `(lhs, rhs)`，`tile.matmul_acc` 为 `(acc, lhs, rhs)`。`lhs` 与 `rhs` 必须是 `Var` / `IterArg`（通过 `AsVarLike` 识别）且为 `TileType`，形状必须是静态 2D。右（B）操作数必须 `memory_space == Mat`（从 DDR 载入 L1 后送入 L0B）；左（A）操作数可以是 `Mat`（QK 模式）**或** `Vec` —— 即 fused-attention 的 `score·V`（PV）模式，softmax/`exp` 的输出在 cube↔vector 边界以 `Vec` 形式到达 matmul。其它情形（Acc 操作数、右操作数为 Vec、动态形状）直接静默跳过。`tile.matmul_bias` 暂不改写——只在最后一次迭代后做 bias-add 需要额外重写，目前尚未实现。
2. **选择 L0 tile 形状** —— 调用 `utils::ChooseL0Tile(cfg)`。`cfg` 来自当前 `BackendHandler` 的 `GetL0{a,b,c}CapacityBytes()` 与 `GetL0FractalAlignment()`，再加上从调用结果类型读出的元素字节宽 `bytes_a/b/c`，使 chooser 看到真实的累加器占用。`c_read = is_matmul_acc`：因为 `tile.matmul_acc` 把调用方的累加器穿过 K-loop iter-arg（chooser 流量模型中 γ_C = 2）。Chooser 返回 `(m, n, k)` —— 闭式 O(1) 算法，依据 L0 切分设计文档（连续最优 + 邻域对齐候选，按 `(traffic, padded_compute, k_blocks, area, k)` 打分）。
3. **若已是 L0 大小则跳过** —— `(m, n, k) == (M, N, K)`。
4. **不支持的形态以 `PerfHint` 跳过**：
   - 子字节 dtype（cube path 不支持）—— `PH-AT-003`。
   - `ChooseL0Tile` 拒绝该配置 —— `PH-AT-005`。
   - `K % k != 0` —— `PH-AT-007`。K 边界处理（最后一次 K 迭代切 `valid_shape`）目前尚未实现；K 切分与 M/N 切分都受此限制。
5. **构造 K-loop**（针对一个输出子块——K 切分时即整个输出，M/N 切分时为每个 `[m, n]` 子块）：
   - `tile.matmul` —— iter-arg 初值为 Acc-resident 的 `tile.create([m, n], dtype, target_memory=Acc)` 占位；循环体用 `IfStmt` 在 `ko == 0` 时走 `tile.matmul`（产生新的 Acc），其它迭代走 `tile.matmul_acc`（向 iter-arg 上累加）。`IfStmt` 物化一个 phi 形式的 `return_var`，由外层 yield 写回 iter-arg。
   - `tile.matmul_acc` —— iter-arg 初值就是调用方传入的累加器（其类型已经与每次迭代的 `tile.matmul_acc` 输出一致）；每次迭代统一是 `tile.matmul_acc`，无需 if-else。
   - 每次迭代的操作数抽取使用 `tile.extract(src, idx_row, idx_col, [shape], target_memory=Left|Right)` —— 这是旧版 `tile.slice`（Mat-resident 中间 tile）+ `tile.mov`（Mat→Left/Right）的 SSA 化合并。这样既消除了 Mat-resident 中间 slice tile，也使得 lower 后是 `pto.textract` 而不是 `pto.subview`，从而绕开后者的 `valid_row` codegen 不一致问题。对于原点为 `(mi, ni)` 的输出子块，抽取的是 `lhs[mi:mi+m, ko:ko+k]` 与 `rhs[ko:ko+k, ni:ni+n]`；K 切分情形即 `mi == ni == 0`、`m == M`、`n == N`。
   - **Vec 左操作数预存（staging）** —— 当左（A）操作数为 `Vec`（PV / `score·V`）时，在 K-loop **之前**插入一次 `tile.move(lhs, target_memory=Mat)`，每次迭代的 Left `tile.extract` 从这个 Mat tile 切片（使抽取源与 QK 路径一样是 Mat）。把 Vec→Mat 这一跨界保持为 `tile.move`，可让 [`ExpandMixedKernel`](22-expand_mixed_kernel.md) 识别它（`CollectCVBoundaryMoves` 只匹配 `tile.move`）并 lower 成跨核 `tpop_from_aiv` 握手（数据落到 Mat）。若直接从 Vec tile 抽取，则会在 cube 侧留下一个悬空的跨界自由变量。
   - K-loop 标记为 `ForKind::Pipeline`，`pipeline_stages=2`。
6. **M/N 切分（当 `m < M` 或 `n < N`）** —— `[M, N]` 输出 Acc 超过 L0c。对于**结果被唯一一个 2D `tile.store(c, base, out)` 消费的普通 `tile.matmul`**，本 pass 把输出切分成 `ceil(M/m) × ceil(N/n)` 的网格：对每个子块原点 `(mi, ni)`，计算该 `[m, n]`（边界处为 `min(m, M-mi) × min(n, N-ni)` 的部分块）子块，并发出 `tile.store(c_sub, [base_r + mi, base_c + ni], out_prev)`。当 **K 跨 ≥ 2 个 L0 块**时，每个子块是独立的**流水化 K-loop**（`[m, K]`/`[K, n]` 操作数面板放不进 L0，需逐子块重新抽取）。当 **`k == K`**（整段 K 一次性放进 L0a/L0b）时，把网格按**嵌套 `ForKind::Pipeline` 循环**发出，覆盖可整除的内部区域 `[0, full_m) × [0, full_n)`（`full_m = ⌊M/m⌋·m`、`full_n = ⌊N/n⌋·n`），使 [`LowerPipelineLoops`](28-lower_pipeline_loops.md) 对移动操作数的 `tile.extract` 做双缓冲（隐藏在 cube 计算之后）。外层循环持有**常驻**面板，按内部区域的总抽取流量择优——A 常驻（行在外）的代价为 `T_row = P·A + P·Q·B`，B 常驻（列在外）为 `T_col = P·Q·A + Q·B`，其中 `P`/`Q` 是内部区域的行/列块数，`A = m·K·bytes_a` / `B = K·n·bytes_b` 是每块面板的抽取字节——该面板每个外层步只重新抽取一次。内层循环为 `pipeline_stages=2` 且 `pipeline_overlap_stores=false`，使 [`CanonicalizeIOOrder`](29-canonicalize_io_order.md) 把每个 store 紧贴其 matmul 排布（只占一块 L0C 累加器，而非两块同时存活）。L 形的**部分边界**（`[full_m, M) × [0, N)` 加 `[0, full_m) × [full_n, N)`）被剥离为直线展开的部分块，因此 `m`/`n` 无需整除 `M`/`N`——不会有把例如 `M = 272 = 16·17` 坍缩成 16×16 子块的整除约束。这些 store 以 SSA 形式串联输出张量；最后一个 store 的结果替换下游对原 store 的引用。以下 M/N 形态**暂未支持**，会发出 `PH-AT-006`（matmul 保持不变）：`tile.matmul_acc`（需对调用方 `[M, N]` 累加器按子块切片）、左操作数为 `Vec`（PV 路径）、以及结果在片上被消费但**并非**完全作为矩阵乘操作数（混合 store + 片上使用，或 elementwise 消费）。结果被**完全作为矩阵乘操作数**消费（链式 matmul）则**不**延后——走下面的 **Mat-scratch** 放置。

   **放置策略（direct-store vs Mat-scratch）。** 两种网格都把每个 `[m, n]` Acc 子块交给一个 `SubtilePlacer`。**`DirectGmPlacer`** 把它 store 到 DDR 输出（上文的 `tile.store`）。**`MatScratchPlacer`** 则把整个 `[M, N]` 结果保留在片上的 L1/**Mat** scratch 中——用 `tile.create(target_memory=Mat)` 创建一次（其隐式 NZ TileView `col_major/row_major` 即矩阵乘操作数布局），随后每个子块通过 `tile.assemble(scratch, sub, [mi, ni])` 就地组装（Acc→Mat，lowering 为 `pto.subview` + `pto.tmov`）。当 matmul 结果的**所有**使用都是矩阵乘操作数读取、**且** `[M, N]` scratch 能放进 backend 的 Mat 容量（`GetMemSize(Mat)`）时，本 pass 才选择 Mat-scratch——这是一个保守的必要条件 gate，把超大的链式 matmul 留在延后的 `PH-AT-006` 路径上，而不是产生一个不可能的片上分配（同时考虑共存 Mat 张量的完整 packed-peak 检查为后续工作）。选中后把结果 `Var` 重映射到 scratch，使消费者在片上读取它。`tile.assemble` 的 `set_output_memory_inherit_input()` 让整条链共享同一个 Mat base，因此组装是就地的（不产生不受支持的 Mat→Mat 保留拷贝）。split-K（展开、常量偏移）与 full-K（流水化、循环变量偏移）网格都可驱动任一 placer。
7. **改写所在 `SeqStmts`** —— 把原 matmul 的 `Var`（K 切分）或消费 store 的结果（M/N 切分）用法改成新的 `return_var`。替换作用域只限当前 `SeqStmts`，不会泄漏到兄弟区域。

本 pass 是 `ProgramPass`，对每个函数走 `IRMutator`；当函数内没有触发任何改写时，返回原函数（不会发生 `MutableCopy` 开销）。

## 示例

### 普通 `tile.matmul`

**Before**（Mat-resident `tile.matmul`，`M = N = 128`，`K = 256`）：

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def main(self, ...):
        ...
        c: pl.Tile[[128, 128], pl.FP32] = pl.tile.matmul(a_mat, b_mat)
        ...
```

**After**（chooser 选定 `m = 128, n = 128, k = 64`）：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def main(self, ...):
        ...
        c_l0_init = pl.tile.create([128, 128], pl.FP32, target_memory=Acc)
        for ko, (c_iter,) in pl.pipeline(0, 256, 64, init_values=(c_l0_init,), stage=2):
            sa = pl.tile.extract(a_mat, 0, ko, [128, 64], target_memory=Left)
            sb = pl.tile.extract(b_mat, ko, 0, [64, 128], target_memory=Right)
            if ko == 0:
                c_first = pl.tile.matmul(sa, sb)
                c_phi = pl.yield_(c_first)
            else:
                c_acc = pl.tile.matmul_acc(c_iter, sa, sb)
                c_phi = pl.yield_(c_acc)
            c = pl.yield_(c_phi)
        # c（即 yield-LHS）持有累加得到的 Acc 类型结果。
        ...
```

### `tile.matmul_acc`

调用方的累加器直接穿过 iter-arg，无需 if-else：

```python
for ko, (c_iter,) in pl.pipeline(0, K, k, init_values=(acc_init,), stage=2):
    sa = pl.tile.extract(a_mat, 0, ko, [m, k], target_memory=Left)
    sb = pl.tile.extract(b_mat, ko, 0, [k, n], target_memory=Right)
    c_new = pl.tile.matmul_acc(c_iter, sa, sb)
    c = pl.yield_(c_new)
# c（即 yield-LHS）持有累加得到的 Acc 类型结果。
```

### M/N 切分（输出超过 L0c）

**Before**（`M = N = 512`，`K = 512`，FP32；`[512, 512]` FP32 输出为 1 MB > L0c，chooser 选 `m = n = 256, k = 32`）：

```python
c: pl.Tile[[512, 512], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
out = pl.store(c, [0, 0], out)
```

**After**（2×2 的 `[256, 256]` Acc 子块网格，每个子块一个流水化 K-loop 并直接 store 到输出——下面只展示一个子块；store 串联为 `out → out_t0 → out_t1 → out_t2 → out_t3`）：

```python
# 子块 (mi=256, ni=0)：行 [256:512]，列 [0:256]。
c_t1_init = pl.tile.create([256, 256], dtype=pl.FP32, target_memory=Acc)
for ko, (c_iter,) in pl.pipeline(0, 512, 32, init_values=(c_t1_init,), stage=2):
    sa = pl.tile.extract(lhs_mat, 256, ko, [256, 32], target_memory=Left)
    sb = pl.tile.extract(rhs_mat, ko, 0, [32, 256], target_memory=Right)
    if ko == 0:
        c_first = pl.tile.matmul(sa, sb)
        c_phi = pl.yield_(c_first)
    else:
        c_acc = pl.tile.matmul_acc(c_iter, sa, sb)
        c_phi = pl.yield_(c_acc)
    c_t1 = pl.yield_(c_phi)
out_t1 = pl.store(c_t1, [256, 0], out_t0)  # 子块 store 到 out[256:512, 0:256]
```

边界子块（当 `m`/`n` 不整除 `M`/`N`）使用静态部分尺寸 `[min(m, M-mi), min(n, N-ni)]` —— 例如 Ascend910B 上的 256×256 FP32 matmul（chooser 选 `m = 192, n = 160`）会切成 `192×160`、`192×96`、`64×160`、`64×96` 四个子块。

## Backend 约束

L0 容量与 fractal 对齐都来自当前 `BackendHandler`。Pass 优先从 `PassContext::Current()->GetBackendHandler()` 读取，若无活动 context 则回退到 `pypto::backend::GetBackend()->GetHandler()`（例如未包 `PassContext` 直接调用的测试场景）。

| Handler 调用 | 用途 |
| ------------ | ---- |
| `GetL0aCapacityBytes()` | chooser 中 L0a (Left) 容量 |
| `GetL0bCapacityBytes()` | chooser 中 L0b (Right) 容量 |
| `GetL0cCapacityBytes()` | chooser 中 L0c (Acc) 容量 |
| `GetL0FractalAlignment()` | chooser 中 M/N/K 对齐粒度 |
| `GetMinL0TileDim()` | 单轴最小 tile 尺寸 |

因此新增 backend 时，只需要提供这些 handler 接口；本 pass 自身与具体 backend 无关。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

**Properties 声明**：`include/pypto/ir/transforms/pass_properties.h`（`kAutoTileMatmulL0Properties`）

**实现**：`src/ir/transforms/auto_tile_matmul_l0_pass.cpp`

**Chooser 工具**：`src/ir/transforms/utils/l0_tile_chooser.cpp` —— 闭式 L0 形状选取，未来其它 tiler 也可复用。

**Python 绑定**：`python/bindings/modules/passes.cpp`

**测试**：`tests/ut/ir/transforms/test_auto_tile_matmul_l0.py`、`tests/ut/ir/transforms/test_l0_tile_chooser.py`

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| Required | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, NormalizedStmtStructure |
| Produced | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, NormalizedStmtStructure |
| Invalidated | — |

## 适用范围

| Op | 处理方式 |
| -- | -------- |
| 静态 2D、右操作数为 Mat（左为 Mat 或 PV 的 Vec）、输出可放进 L0c 的 `tile.matmul` | 改写为 2 阶段流水化 K-loop；Vec 左操作数先预存到 Mat |
| 输出超过 L0c、被唯一一个 2D `tile.store` 消费的普通 `tile.matmul`（左右均 Mat） | M/N 切分：`ceil(M/m) × ceil(N/n)` 子块网格，每个子块一个 K-loop 并直接 store 到输出（direct-store） |
| 输出超过 L0c、被**完全作为矩阵乘操作数**消费（链式 matmul）、且 `[M, N]` scratch 能放进 Mat/L1 的普通 `tile.matmul` | M/N 切分到 L1/**Mat** scratch（逐子块 Acc→Mat `tile.assemble`），保留在片上供消费者读取（Mat-scratch） |
| 静态 2D、右操作数为 Mat（左为 Mat 或 PV 的 Vec）、输出可放进 L0c 的 `tile.matmul_acc` | 改写为 2 阶段流水化 K-loop（循环体统一为 `matmul_acc`） |
| 右（B）操作数为 Vec 的 `tile.matmul[_acc]` | 跳过（B 操作数必须从 L1 送入 L0B） |
| `tile.matmul_bias` | 跳过（待支持——「最后一次迭代后再 bias-add」的改写尚未实现） |
| 已经是 L0 大小（`(m, n, k) == (M, N, K)`）的 matmul | 不动 |
| 输出超过 L0c 但两种 M/N 放置都不适用——`matmul_acc`、Vec 左操作数、非矩阵乘操作数消费者、或 `[M, N]` 超过 Mat/L1 的链式 matmul scratch | 以 `PerfHint`（`PH-AT-006`）跳过 |
| 子字节 dtype / `K % k != 0` | 以 `PerfHint` 跳过 |
| 非 InCore 函数（Orchestration、Opaque） | 不动 |

## Diagnostics

当 pass 决定不改写时，会发出 `PerfHint`（而不是失败）；原 matmul 保持不变并继续走后续流水线。`PerfHint` 编码：

| 编码 | 含义 |
| ---- | ---- |
| `PH-AT-003` | 操作数或累加器使用了子字节 dtype |
| `PH-AT-005` | `ChooseL0Tile` 拒绝了该配置 |
| `PH-AT-006` | 输出超过 L0c，但两种 M/N 放置都不适用——`tile.matmul_acc`、左操作数为 Vec、或结果在片上被消费但**并非**完全作为矩阵乘操作数（混合 store + 片上、或 elementwise）。结果被完全作为矩阵乘操作数消费时走 **Mat-scratch** 路径（不发提示）——但若其 `[M, N]` scratch 超过 backend 的 Mat/L1 容量，则同样在此延后（保守的必要条件 gate；完整的 packed-peak 检查为后续工作）。 |
| `PH-AT-007` | `K % k != 0`（K 边界处理暂不支持） |
| `PH-AT-008` | `ChooseL0Tile` 返回了 fallback 配置并附带 perf hint |

## 相关 Pass

- [`FlattenTileNdTo2D`](15-flatten_tile_nd_to_2d.md) —— 上游 pass；产生本 pass 所需的静态 2D Mat-resident tile 形状
- [`InferTileMemorySpace`](18-infer_tile_memory_space.md) —— 下游 pass；负责桥接本 pass 故意保留下来的 Vec/Acc 累加器
- [`LowerPipelineLoops`](28-lower_pipeline_loops.md) —— 消费本 pass 产生的 `ForKind::Pipeline` + `pipeline_stages=2`
