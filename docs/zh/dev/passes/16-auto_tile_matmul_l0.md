# AutoTileMatmulL0 Pass

针对静态 2D `tile.matmul`、`tile.matmul_acc` 与 `tile.matmul_bias` 进行 L0 切分：从当前 backend 的 L0 容量中挑选 L0 tile 形状 `(m, n, k)`，并把调用改写成一个 2 阶段流水化的 K-loop，每个迭代用 `tile.extract` 从 Mat 抽取 Left/Right 操作数。容量检查采用累加器在 L0C 中的物理占用，而不只是逻辑形状。该物理占用超过 L0c 时，本 pass 会对新的普通或带 bias 的 matmul，或前端规范的 split-K create/pipeline/store 链做 M/N 切分。

## 概览

由 `ConvertTensorToTileOps` + [`FlattenTileNdTo2D`](13-flatten_tile_nd_to_2d.md) 生成的 Mat-resident matmul 通常带有完整的 `(M, N, K)` 操作数形状——几乎一定大于 cube unit 的 L0a/L0b/L0c 容量。本 pass 选取一个能放进 L0 的 `(m, n, k)`，并把该 matmul 改写成一个 K-loop：循环体内用 `tile.extract` 把 `[m, k]` 与 `[k, n]` 的切片送入 `Left` / `Right`，并把累加器写入 `Acc`-resident 的 iter-arg。该循环带有 `ForKind::Pipeline` 与 `pipeline_stages=2`，使下游 [`LowerPipelineLoops`](29-lower_pipeline_loops.md) 可对每次迭代的操作数 `tile.extract` 生成 2 级 ping-pong。

已经放入 `Left` 或 `Right` 的操作数表示程序员手工做出的 L0 调度决策，因此 AutoTile 不会静默替换或再次切分它；合法的手工 tile 保持不变。若一个静态操作数自身就超过 backend 对应空间的容量——例如 FP16 `Right[256, 256]` 需要 131072 字节，而 L0B 只有 65536 字节——本 pass 会在该操作数处报错，给出变量名、物理形状、dtype、所需/可用字节数以及两种修复方式：让操作数保留在 `Mat` 并直接传给 matmul，由 AutoTile 选择合法的 L0 tile；或手工抽取更小的 L0 tile。这个更早、面向具体算子的错误也避免了随后出现无关的 MemoryReuse 回退 warning。

本 pass 在 PyPTO 内存规划器下也处理已经切好 L0 tile 的形式：用户编写的静态 `pl.pipeline(stage=F)`（`F ≥ 2`，且迭代数能被 `F` 整除），每次迭代恰有一个 `tile.matmul(Left, Right)`，且具有一条规范的循环携带 drain 链。选中的移动操作数必须由每次迭代中直接的 Mat→L0 传输产生，另一个矩阵乘操作数则定义在循环外。只有当对应 drain 路径的盈利性门限通过，且整个函数中 Acc 的保守占用量（包括 pipeline lowering 可能为其他 Acc 生产者请求的物理 stage 副本数）再加上每个合格循环的一块额外 slot 仍能放入 L0C 时，AutoTile 才启用两槽 L0C ping-pong。Direct-to-GM `tile.store` 至少需要四次迭代；更便宜的 Acc→Mat `tile.assemble` 路径至少需要八次迭代，且按分配规则对齐后的单块 Acc 必须至少占 L0C 的四分之一。在 128 KiB L0C 上，该门限会接纳两组独立实测获益的 32/40 KiB Mat-scratch case，同时排除实测回退的 8 KiB case 和打平的 16 KiB case。Pipeline lowering 随后按两级一组发射 `matmul, matmul, drain, drain`，使 tile *i* 的 FIXPIPE drain 与 tile *i+1* 的 MAD 重叠。更深的操作数流水线保留原有 stage membership（仍受分配器常规容量门限约束），而 Acc membership 在两块 slot 间轮转。若存在多个 Acc、额外 store、嵌套控制流、间接使用、循环携带的矩阵操作数、单独 lowering 的尾组，或非规范的 drain/yield 链，则保持不变。PTOAS 保持原样，因为其规划器已为复现循环分离物理 Acc，且设备计时未显示该源级标记带来收益。

**K 切分 vs M/N 切分。** 当 chooser 返回 `m == M` 且 `n == N` 时，输出的**物理**分配能放进 L0c，因此只切分 K 维（一个 K-loop）。常规容量按 `AlignUp(M, GetL0cMAlignment(dtype)) × N × bytes_c` 计算；例如 Ascend910B 上逻辑 `M = 16` 的 INT32 累加器会占用 32 个物理行。规范 split-K 路径还会把操作数布局的 Mat box 粒度传给 chooser，在选择 tile 前按 `AlignUp(AlignUp(m, box_m), l0c_align_m) × AlignUp(n, box_n) × bytes_c` 计入容量，并同样按补齐后的尺寸检查 L0A/L0B。当返回 `m < M` 或 `n < N` 时，输出会超过 L0c。由于操作数已经是 Mat-resident，*只有*输出溢出：本 pass 把**输出**切成 `ceil(M/m) × ceil(N/n)` 的 `[m, n]` 子块网格（边界处为部分块——`m`/`n` 不必整除 `M`/`N`）。新的普通或带 bias 的 matmul 直接逐子块计算和放置；带 bias 的 matmul 会把仅使用一次、且到 matmul 之间只有同层 load 的完整 bias load，替换为同一无 effect 观测区间内的 `[1,n]` 窗口 load，并且只在每个输出块的第一个 K block 上加一次 bias。对于前端规范的 split-K 形式，则为每个输出子块克隆完整的源 K 归约，使一块 `[m, n]` 累加器完成全部 K block 并 store 后，才开始下一块。因而不会实例化超大的完整 `[M, N]` Acc。输出张量以 SSA 形式在各子块 store 间串联（`out → out_t0 → out_t1 → …`）。

**统一的物理大小契约。** `ChooseL0Tile`、已有流水线的 dbC 容量规划与 `InitMemRef` 共同使用同一个 backend-aware L0C 行占用 helper。规范 split-K 还把共享的 layout-aware Mat box 对齐传给 `ChooseL0Tile`，因此选择阶段会先应用与 load 重建一致的 M/N 补齐，再应用统一的 L0C 行补齐。chooser 或 dbC 规划接纳的 tile 会按容量判断使用的补齐形状分配；两个同时存活的补齐累加器不会被放到逻辑上相邻、但物理上重叠的地址区间。

**Fits-L0c 链式 cast-fold（cast 折叠）。** 当链式 matmul 的 `[M, N]` 结果*能放进* L0c（无需 M/N 切分），但经一次降精度后再喂给第二个 matmul —— `c = matmul(a, b); cb = cast(c, bf16); d = matmul(cb, e)` —— 消费者需要 bf16 中间值位于 **Mat**（L1）。若不处理，`tile.cast` 会 lower 成 **Vector** 的 `pto.tcvt`（一次 cube→vector→cube 往返，在 `[128, 128]` 形状下会撑爆 Vec buffer）。本 pass 改为把 cast 折叠成**一次整窗**的 Acc→Mat `tile.assemble` —— 与超大 Mat-scratch 路径用的是同一个 `MatScratchPlacer`，只是单次 `PlaceAt` 于偏移 `(0, 0)` 而非一个网格 —— 从而让降精度留在 cube 上，作为 FIXPIPE 的 `pto.tinsert`。这是一个与 K 切分无关的 cast-peephole：无论 producer 是保持整体（`k == K`）还是被 K-loop 切分（`k < K`）都会触发，且仅当 cast 结果的每一处使用都是矩阵乘操作数时才折叠（非矩阵乘消费者保留 Vector cast）。折叠还严格对齐 FIXPIPE 能复现的能力——即 **`f32 → bf16/f16`** 降精度、且舍入模式为 **`rint`**（就近、**取偶**），这是 FIXPIPE 固定的 tie 规则——A2/A3 与 A5 一致（pto-isa 的 CPU 参考实现用 `std::bfloat16_t` 降精度、无 arch 分支，且 `pto.tinsert` 不带 `rmode`；两个 backend 仅 scratch dtype 不同，舍入相同）。若源不是 `f32`（例如 `int32` 矩阵乘结果，需要带 scale 的 *dequant*）、为 cast 默认的 **`round`** 模式（就近、**远离零**），或为有方向/截断的模式（`none`/`floor`/`ceil`/`trunc`/`odd`），则都保留 Vector `pto.tcvt`——只有它才会遵循所请求的 `rmode`——并由本 pass 发出指向 `mode="rint"` 的 `PH-AT-010` 提示。同一道 gate（`CastFoldableToFixpipeMat`）也用于下面的超大 Mat-scratch 折叠。超大结果不会到达这个 peephole——它们的 cast 由上面的 M/N 路径逐子块折叠。

**Pipeline 位置**：紧跟在 [`LegalizeTileCast`](15-legalize_tile_cast.md) 之后，先于 [`CanonicalizeTileSlice`](17-canonicalize_tile_slice.md) 与 [`InferTileMemorySpace`](18-infer_tile_memory_space.md)。此时 tile op 已是 2D，但 memory space 尚未推断。

**前置属性 (Required)**：`SSAForm`、`SplitIncoreOrch`、`IncoreTileOps`、`TileOps2D`、`NormalizedStmtStructure`。

**产出属性 (Produced)**：与前置属性相同（属性保持不变的改写）。

**失效属性 (Invalidated)**：无。

**何时使用**：一律在默认 tile 阶段流水线中运行。如果没有需要切分或折叠 cast 的 Mat-resident matmul，且没有符合自动累加器双缓冲条件的、由 PyPTO 规划的已有 L0 流水线，本 pass 是 no-op。

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

对每个 InCore 函数中的 `tile.matmul`、`tile.matmul_acc` 或 `tile.matmul_bias`：

1. **过滤** —— 操作数布局：`tile.matmul` 为 `(lhs, rhs)`，`tile.matmul_acc` 为 `(acc, lhs, rhs)` 或 `(acc, lhs, rhs, init_cond)`，`tile.matmul_bias` 为 `(lhs, rhs, bias)`。4 操作数形式**仅**对累加形态成立 —— 普通 matmul 没有可被谓词化的累加器，`tile.matmul_bias` 也没有 `init_cond` 操作数。所有矩阵操作数必须是 `Var` / `IterArg`（通过 `AsVarLike` 识别）且为静态 2D `TileType`。在 residency 过滤之前，若静态矩阵操作数已位于最终的 `Left` / `Right` 空间，且其自身物理占用超过对应 L0A/L0B 容量，则立即拒绝；这种不可能的手工分配无法通过复用或更换 planner 修复。自身合法的手工 L0 操作数仍保持不变。对于自动 tiling，右（B）操作数必须是 `Mat`。普通/累加 matmul 保留已有的 Mat 或 Vec 左操作数支持；带 bias 的 matmul 刻意要求两个矩阵操作数均为 `Mat`，bias 必须为 `[1, N]`、位于 `Mat` 或 `Bias`，并使用累加器数据类型（浮点矩阵操作数对应 FP32，整数矩阵操作数对应 INT32）。Mat-resident bias 还要求 backend 支持对应的 Mat→Bias dtype 组合；带 bias 的 M/N/K 必须满足矩阵操作数 layout 推导出的 box 对齐，从而每个生成的 Left/Right/Bias 物理 box 都合法。其它情形会被保守跳过或诊断。
2. **选择 L0 tile 形状** —— 调用 `utils::ChooseL0Tile(cfg)`。`cfg` 来自当前 `BackendHandler` 的 `GetL0{a,b,c}CapacityBytes()`、`GetL0FractalAlignment()` / `GetMinL0TileDim()`、`GetL0cMAlignment(accumulator_dtype)` 以及 `GetL0CostModel()`（L1↔L0 带宽 + MAD 发射开销），再加上从调用结果类型读出的元素字节宽 `bytes_a/b/c`。带 bias 的 matmul 还会用 `GetBiasCapacityBytes() / bytes_c` 限制候选 N；这是辅助 SRAM 的硬上限，不是 roofline 系数。对于在 full-K 输出网格内重建的 Mat-resident bias，同一次穷举搜索会按调度应用不同上限：B-stationary 只需一个 Bias slot；A-stationary 与 N 位于外层的 output-stationary 需要两个；N 位于内层、同时受两层 pipeline membership 约束的 output-stationary 需要四个。已经位于 Bias 的源定义在网格外，始终只占一个 slot。L0C 候选合法性为 `AlignUp(AlignUp(m, box_align_m), l0c_align_m) × AlignUp(n, box_align_n) × bytes_c × dbC <= L0C`；box 对齐默认是 1，规范 split-K 路径会在物理补齐窗口时从操作数的有效 Mat layout 设置它们。补齐后的 M/N 同时用于 L0A/L0B 容量检查，而逻辑计算形状与 roofline 计费仍为 `[m, n]`。`c_read = is_matmul_acc`：因为 `tile.matmul_acc` 把调用方的累加器穿过 K-loop iter-arg（γ_C = 2，使模型计入的 C 流量翻倍）。Chooser 返回 `(m, n, k)` 以及所选的设计点（design point）—— 这是对 roofline `wall` 的**穷举最小化**，并非闭式解；详见下文 [Cost model & design space](#cost-model--design-space-choosel0tile)。
3. **若已是 L0 大小则跳过** —— `(m, n, k) == (M, N, K)`。
4. **不支持的形态以 `PerfHint` 跳过**：
   - 子字节 dtype（cube path 不支持）—— `PH-AT-003`。
   - `ChooseL0Tile` 拒绝该配置 —— `PH-AT-005`。
5. **构造 K-loop**（针对一个输出子块——K 切分时即整个输出，M/N 切分时为每个 `[m, n]` 子块）：
   - `tile.matmul` —— iter-arg 初值为 Acc-resident 的 `tile.create([m, n], dtype, target_memory=Acc)` 种子；循环体是单条带谓词的 `tile.matmul_acc(c_iter, sa, sb, ko == 0)`，谓词在第一个 L0 block 上覆写累加器，之后各 block 累加。由于 `tile.matmul_acc` 声明了 `set_output_reuses_input(0)`，create / call / yield / return_var 这条链在构造上就落在同一块 L0C buffer 上。它替换掉的 `if ko == 0` 剥离形态会把同一个逻辑值放在**两块** Acc buffer 上再在 phi 处合并——这是任何受支持目标都无法实现的状态：除 FIXPIPE 排空外没有任何单元读 L0C，因此不存在可用来调和两个分支的 Acc→Acc 拷贝。
   - `tile.matmul_acc` —— iter-arg 初值就是调用方传入的累加器（其类型已经与每次迭代的 `tile.matmul_acc` 输出一致）；每次迭代统一是 `tile.matmul_acc`。3 操作数形式完全不带谓词，因为该累加器在第一次迭代时就已经有效，绝不能被覆写。4 操作数形式携带调用方的 `init_cond`，其含义是「这是*用户*归约的第一个 K 步」—— 生成的调用把它与循环自身的 `ko == 0`（「该步的第一个 L0 block」）做与运算，因此只有两者同时成立时才覆写累加器。当只有一个直线完整块（`⌊K/k⌋ == 1`）时，`init_cond` 原样透传：该块*就是*第一个 K block，再与静态为真的 `ko == 0` 组合只会多出一个可折叠节点。剥离出的部分尾块则完全丢弃谓词、发出 3 操作数调用 —— 它运行在所有完整块之后的 K 偏移上，永远不是第一个块。
   - `tile.matmul_bias` —— 第一个 K block 被**提到循环之外**（head-peel）：一条直线展开的 `tile.matmul_bias` 恰好只加一次 bias *并*铸造出累加器，随后循环在其余整块上用普通 `tile.matmul_acc` 累加进去。`tile.matmul_bias` 没有 `init_cond` 操作数，无法使用普通 `tile.matmul` 所用的带谓词循环体；提取第一个 block 可以在没有谓词的情况下得到同样的单 buffer 链。它刻意**不**剥离成 `IfStmt`：在新建的 `tile.matmul_bias` 与原地的 `tile.matmul_acc` 之间放置 phi，会让同一个逻辑值拥有两个位于不同 L0C buffer 的生产者，这是任何 target 都无法实现的（不存在 `Acc`→`Acc` 拷贝），`MemoryReuse` 只能事后改写 buffer 身份来修复。整块恰好为两个时，第二个也走直线展开，因为 1 次迭代的流水循环是退化的。因此本 pass 完全不再生成累加器 phi。N 切分时，对应窗口从原始 tensor 重新 `tile.load` 到独立 Mat tile，再通过 `tile.move` 传到 Bias（`pto.tload` + `pto.tmov`）。不会使用单行 Mat `tile.slice`，因为其 boxed `pto.subview` 不满足 PTOAS 合法性。
   - 每次迭代的操作数抽取使用 `tile.extract(src, idx_row, idx_col, [shape], target_memory=Left|Right)` —— 这是旧版 `tile.slice`（Mat-resident 中间 tile）+ `tile.mov`（Mat→Left/Right）的 SSA 化合并。这样既消除了 Mat-resident 中间 slice tile，也使得 lower 后是 `pto.textract` 而不是 `pto.subview`，从而绕开后者的 `valid_row` codegen 不一致问题。对于原点为 `(mi, ni)` 的输出子块，抽取的是 `lhs[mi:mi+m, ko:ko+k]` 与 `rhs[ko:ko+k, ni:ni+n]`；K 切分情形即 `mi == ni == 0`、`m == M`、`n == N`。
   - **Vec 左操作数预存（staging）** —— 当左（A）操作数为 `Vec`（PV / `score·V`）时，在 K-loop **之前**插入一次 `tile.move(lhs, target_memory=Mat)`，每次迭代的 Left `tile.extract` 从这个 Mat tile 切片（使抽取源与 QK 路径一样是 Mat）。把 Vec→Mat 这一跨界保持为 `tile.move`，可让 [`ExpandMixedKernel`](22-expand_mixed_kernel.md) 识别它（`CollectCVBoundaryMoves` 只匹配 `tile.move`）并 lower 成跨核 `tpop_from_aiv` 握手（数据落到 Mat）。若直接从 Vec tile 抽取，则会在 cube 侧留下一个悬空的跨界自由变量。
   - K-loop 标记为 `ForKind::Pipeline`，`pipeline_stages=2`。
   - **非整除 K（K 边界剥离）** —— 当所选 `k` 不整除 `K` 时，流水化循环只覆盖 `⌊K/k⌋` 个完整块（上界 `⌊K/k⌋·k`），再用一个直线展开的 `tile.matmul_acc` 剥离宽度为 `K − ⌊K/k⌋·k` 的部分尾块；当只有一个完整块（`⌊K/k⌋ == 1`）时，用「单个直线完整块 + 尾块」替代循环。`K` 与 `k` 均为 16 对齐（cube 分形），故剥离出的尾块宽度 `K − ⌊K/k⌋·k` 本身也是 16 对齐——一个普通的 `matmul_acc` 块，无需掩码。（ptoas 要求 tile 列数为 16 的倍数，故操作数维度必须 16 对齐；**不支持**非 16 对齐的 `K`。）chooser 仅在 `ChooseL0Tile` 的 `allow_k_boundary`（本 pass 已开启）下返回非整除 `k`；当整段（16 对齐的）K 能放进一个 L0 块时，chooser 返回 `k == K`（无循环）。**非 16 对齐的 `K` 会被直接拒绝**——不存在合法的 K 切分（任何剥离尾块或整段 K 块的列数都非分形），故 chooser 不返回任何候选，本 pass 以 `PH-AT-007` 提示跳过该 matmul，而非发出非法的 extract。
6. **M/N 切分（当 `m < M` 或 `n < N`）** —— `[M, N]` 输出 Acc 的物理占用超过 L0c。

   对于**结果被唯一一个 2D `tile.store(c, base, out)` 消费的新 `tile.matmul` 或 `tile.matmul_bias`**，且该消费 store 是 matmul 之后第一条非 load 语句时，本 pass 把输出切分成 `ceil(M/m) × ceil(N/n)` 的网格：对每个子块原点 `(mi, ni)`，计算该 `[m, n]`（边界处为 `min(m, M-mi) × min(n, N-ni)` 的部分块）子块，并发出 `tile.store(c_sub, [base_r + mi, base_c + ni], out_prev)`。带 bias 的 matmul 还会从定义 tensor 将对应的 `[1, n]` 窗口重新 load 到 Mat，再 move 到 Bias，由 cube 将它广播到该子块的 M 行。要求 store 是第一条非 load 语句，可防止延后发射的网格跨越有副作用的操作。当 K 跨多个 L0 块时，每个子块使用独立的流水化 K-loop；当 `k == K` 时，则在可整除内部区域上发出嵌套循环，使 [`LowerPipelineLoops`](29-lower_pipeline_loops.md) 双缓冲移动操作数。外层循环持有常驻面板，output-stationary 或 A/B-stationary 的循环序遵循 chooser 的设计点。L 形边界被剥离为直线展开的部分块，因此 `m`/`n` 无需整除 `M`/`N`。这些 store 以 SSA 形式串联输出张量；最后一个 store 的结果替换下游对原 store 的引用。

   **前端规范的 split-K 归约**被匹配为三个相邻部分：一个 `tile.create([M, N])` 全输出累加器占位值；一个循环携带该值、完成 K 归约的 pipeline；以及一个 2D 输出 store。其中的归约本身接受**两种源写法**：

   - **peel 写法** —— 一个 `if`，首块使用 `tile.matmul`、后续块使用 `tile.matmul_acc`，在分支的 phi 处汇合；
   - **谓词写法** —— 单个 4 操作数 `tile.matmul_acc(acc, lhs, rhs, init_cond)` 累加到循环唯一的 iter-arg，没有分支也没有 phi。

   谓词写法中，`init_cond` 必须是 split-K 的首块判定：`<x> == 0`，其中 `x` 是该循环自身的归纳变量，或者是循环体内定义为 `<归纳变量> * <非零常量>` 的标量（即所有 peel 写法在 `if` 中已经使用的 `k0 = kb * K_TILE`）。按指针身份匹配归纳变量，正是用来拒绝调用方传入的标志位或另一个循环的变量——两者都不能证明累加器会在第一个 K block 上被覆写。这样的三元组会保持不变，并以 `PH-AT-006` 提示报告，而不是无声地失去 M/N 切分。

   两个不同的操作数必须来自循环内直接的 GM→Mat load，且静态 shape 与 valid shape 都覆盖完整矩形面板；除此之外循环只能含标量地址计算。本 pass 把输出网格移到源 K-loop 外：对每个 `(mi, ni)` 创建合法的 `[m, n]` Acc，克隆完整 K-loop，把两次 load 收窄到该输出窗口，完成全部 K 归约后再 store。改写后的循环保留源使用的写法，因此两种写法产生相同的输出网格、相同的收窄 load 和相同的 store 链，只在那一条归约语句上不同。随后普通的调用级 AutoTile 改写会继续对收窄后的调用做必要的内部 K 切分——一旦触发，peel 写法会在分支后产生*两个*内层 K-loop，而谓词写法只产生一个，其谓词为 `init_cond and ko == 0`。该顺序不可颠倒：只切片已有的完整 Acc 仍然需要不可能的 `[M, N]` L0C 分配。

   以下 M/N 形态仍然**暂不支持**：带调用方自有累加器、且不匹配上述规范链的任意独立 `tile.matmul_acc`；左操作数为 `Vec`（PV 路径）；带 bias 的 matmul 已位于 `Bias` 的源还需要不受支持的 Bias-to-Bias N 子窗口；Mat bias 不是仅使用一次、且到 matmul 之间只有同层 load 的 2D load；以及结果在片上被消费但并非完全作为矩阵乘操作数。把所有非 load 语句当作 barrier 会保留跨越 store 或其它 effect 时的原始 bias snapshot 语义，删除被替换的完整 load 则避免冗余流量。结果被完全作为矩阵乘操作数消费时走下面的 Mat-scratch 放置。普通路径使用 `PH-AT-006`，bias 窗口限制使用 `PH-AT-011`。

   **放置策略（direct-store vs Mat-scratch）。** 两种网格都把每个 `[m, n]` Acc 子块交给一个 `SubtilePlacer`。**`DirectGmPlacer`** 把它 store 到 DDR 输出（上文的 `tile.store`）。**`MatScratchPlacer`** 则把整个 `[M, N]` 结果保留在片上的 L1/**Mat** scratch 中——用 `tile.create(target_memory=Mat)` 创建一次（其隐式 NZ TileView `col_major/row_major` 即矩阵乘操作数布局），随后每个子块通过 `tile.assemble(scratch, sub, [mi, ni])` 就地组装。链式路径使用的 bf16/f16 低精度 scratch 会把该 Acc→Mat 回写 lower 成 FIXPIPE `pto.tinsert`；受支持的同 dtype 整窗 assemble 则使用 `pto.subview` + `pto.tmov`。当 matmul 结果的**所有**使用都是矩阵乘操作数读取、**且** `[M, N]` scratch 能放进 backend handler 的 Mat 容量（`GetMatCapacityBytes()`）时，本 pass 才选择 Mat-scratch——这是一个保守的必要条件 gate，把超大的链式 matmul 留在延后的 `PH-AT-006` 路径上，而不是产生一个不可能的片上分配（同时考虑共存 Mat 张量的完整 packed-peak 检查为后续工作）。选中后把结果 `Var` 重映射到 scratch，使消费者在片上读取它。`tile.assemble` 的 `set_output_memory_inherit_input()` 让整条链共享同一个 Mat base，因此组装是就地的（不产生不受支持的 Mat→Mat 保留拷贝）。split-K（展开、常量偏移）与 full-K（流水化、循环变量偏移）网格都可驱动任一 placer。

   > **旧版 PYPTO planner 的限制 —— operand-stationary 链式生产者 + L0 打包。** 链式 matmul（Mat-scratch）的生产者与其消费者共享 L0（顺序执行；中间结果留在 L1，绝不经 DDR —— `L0C→L1→L0A` 往返）。A/B-stationary 的生产者钉住一块占满 L0 的整块操作数缓冲，而流水化消费者需要多块较小缓冲。旧版 PYPTO planner 的 `AllocateMemoryAddr` 只是把复用类顺序堆叠、从不细分已释放区域（例如，一块 64 KB 生产者缓冲被复用给一块 32 KB 消费者 slot 会浪费 32 KB，另一块 slot 溢出 → L0 超限），因此本 pass 仅在该 planner 下强制 Mat-scratch 生产者使用 **output-stationary**。启用实验性 dbC opt-in 后，某些 output-stationary 链仍会暴露该限制；当前一个复现会在 64 KB L0B 上请求 96 KB。`DSA_RP` 与 `PTOAS` 已按实际生命周期放置缓冲，可保留 chooser 选出的 operand-stationary 调度，并把消费者的较小缓冲打包到已释放的整块范围中。为旧版 allocator 增加等价的按生命周期细分能力仍由 [issue #1908](https://github.com/hw-native-sys/pypto/issues/1908) 跟踪。
7. **改写所在 `SeqStmts`** —— 把原 matmul 的 `Var`（K 切分）或消费 store 的结果（M/N 切分）用法改成新的 `return_var`。替换作用域只限当前 `SeqStmts`，不会泄漏到兄弟区域。

8. **识别已有的 L0 流水线** —— 独立于 chooser 驱动的改写，检查每个由 PyPTO 规划、静态、`pipeline_stages=F ≥ 2` 且迭代数能被 `F` 整除的 `ForKind::Pipeline`。要求完整 stage 组可避免单独 lowering 的尾组需要额外 Acc slot。其平坦循环体必须恰有一个普通 `tile.matmul` 和静态 `Left`/`Right` 操作数；选中的移动操作数必须有一个可识别、直接的每迭代 Mat→L0 生产者，而固定操作数定义在循环外。循环体还必须包含一条规范 drain 链，其结果需通过匹配的 iter-arg yield 回去：direct-to-GM `tile.store(acc, ..., iter_arg_i)` 或 Acc→Mat `tile.assemble(iter_arg_i, acc, ...)`。Direct 路径至少需要四次迭代；Mat-scratch 路径至少需要八次迭代，且按分配规则对齐后的 Acc 占用至少为 `ceil(L0C/4)`。后者是独立的保守门限，因为共享的 direct-GM roofline 尚未表达其更便宜的 drain。存在任何其他 Acc 定义/读取或 store-like 操作时都不处理该循环。附加 `pipeline_double_buffer_c=true` 与 `pipeline_overlap_stores=false` 前，本 pass 会保守求和函数中的每个静态 Acc 值。普通 cube 累加器因 lowering 将其串行化而只计一份；其他 Acc 生产者则按其所有外层源 pipeline stage 深度的乘积计数，与 lowering 可能请求的物理 membership 数一致。随后为每个盈利循环增加一块按分配规则对齐的 slot。只有该 lowering 后上界能放入 L0C 时才同时启用这些循环，从而避免 dbC 因漏计其他流水线复制的 Acc 占用而迫使其降低 buffering depth。已有显式属性的循环保持不变。对于 `F > 2`，lowering 重复发射两级 `MMSS` 分组，并把 Acc membership 对 2 取模，而操作数 membership 仍保留深度 `F`。

本 pass 是 `ProgramPass`，对每个函数走 `IRMutator`；当函数内没有触发任何改写时，返回原函数（不会发生 `MutableCopy` 开销）。

## Cost model & design space (`ChooseL0Tile`)

`ChooseL0Tile` 通过**穷举式 roofline 搜索**挑选 L0 GEMM tile，而非闭式公式。对每个合法且对齐的 `(m, n, k)`（每维都是 `GetL0FractalAlignment()` 的倍数，L0C 预算按 `AlignUp(m, l0c_align_m) × n` 计算），它以核心 cycle 估算 wall-clock 并返回最小者：

- 当 FIXPIPE 的 L0C→L1 drain 暴露在外（单 L0C）时，`wall ≈ max(C_load, C_mad) + C_drain`；
- 当 drain 被计算掩盖（L0C 双缓冲，`T` 个输出 tile）时，`wall ≈ max(C_load, C_mad, C_drain) + min(compute, C_drain) / T`。其中 `+ min(…)/T` 是流水线的**填充/排空气泡**——第一个 tile 的计算（或最后一个 tile 的 drain）没有可重叠的对象，因此理想的全掩盖 `T·max` roofline 会少算一个 tile 的非主导流水（两个输出 tile 时为较小流水的 50%，在 2×2 网格上约为 25%）。这可避免在小网格上过度选择 dbC=2。

`C_load` 是所选循环序下 L1→L0A/L0B 的操作数流量，按 `GetL0CostModel()` 给出的各 buffer 带宽缩放（设备 MTE1 实测：`bw_l0a≈130`、`bw_l0b≈85` B/cyc，约 1.52:1）；`C_mad` 是 cube MAD 代价（每条 `TMATMUL` 的发射开销 × K-fractal 数）。`C_drain` 是 FIXPIPE 的 L0C 回写，**按每个输出 tile 计费**、且为**按 M-行**的代价：`⌈M/m⌉·⌈N/n⌉ · (drain_fixed + m·(max(drain_row, bytes_c·n/bw_drain) + drain_penalty·(odd(⌈n/N0⌉)−1)))`。这是对设备 FIXPIPE 实测的直接拟合：FIXPIPE 每次只处理 `N1 M1 M0 N0` FRACTAL_NZ 累加器的一个 M-行（故代价 ∝ `m`），每行用分组 `nburst`/`loop` 遍历 `N1 = ⌈n/N0⌉` 个 N-fractal（`N0 = 32/bytes_c = 8`，fp32 L0C）。每行代价是 `max(floor, throughput)`——一个与 N 无关的固定 burst-issue **下限** `drain_row`（窄 N 时主导），或按字节的 **吞吐** `bytes_c·n/bw_drain`（宽 N 时主导，交叉点约 n=131）——再加**非对齐**残差：非 2 的幂的 fractal 数会把奇部 `odd(N1)−1` 串行成额外 pass，每 M-行按 `drain_penalty` 计费（判据是 **`N1` 非 2 的幂**，而非字面的 `N%32`：`n=80 → odd(10)=5` 被惩罚，`n=96 → odd(12)=3` 也被惩罚，尽管 `96%32=0`；对齐的 2 的幂 `N1`（如 `n=128 → 16`）不计费）。由于 drain 数为 `⌈M/m⌉·⌈N/n⌉`，**拆分输出（M/N）会增加 drain 数，而拆分 K 不会**（部分和在单块 L0C 上累加，每个 `(m,n)` 块只回写一次）。按-M-行的形式使 chooser 倾向**宽-N / 小-M** 的 tile（每次 drain 的 FIXPIPE 行更少），并把非对齐-N tile 正确定价从而不被过度选择——例如 `320×320` 选到对齐的 `(160,128,64)`，而非 drain-bound 的 `160×80`。设备验证（drain 0.93–1.09×，load R²=0.993）。搜索对每个 `(m, n)` 的**所有**合法 `k` 都穷举（不是只取最大合法 k —— 当 `kt ≠ align_k` 时 `⌈K/k⌉·⌈k/kt⌉` 关于 `k` 非单调）。wall 平局时按 `(padded_compute, ⌈K/k⌉, C_load, …)` 字典序决出；其中 `C_load` 键在 MAD-bound 的 `(m,n)`↔`(n,m)` 平局中挑出隐藏 load 更低的那一侧（L0B 带宽更慢，故 m-block 更少者更省）。

搜索覆盖**设计空间（design space）** `P = (m, n, k, stationarity, dbC)`：

- **stationarity（常驻方向）** `{output, A, B}` —— 哪个操作数在 L0 网格上被钉住（常驻）。它**推导出**各操作数的双缓冲深度（`dbA`/`dbB`）：移动的操作数双缓冲（深度 2），常驻的单缓冲（深度 1）。它们不被独立搜索。
- **dbC** `{1, 2}` —— 是否对 L0C 累加器做双缓冲，以便把 FIXPIPE drain 与下一个 tile 的计算重叠。

一个**可实现掩码（realizable mask）**（即 `allow_a_stationary` /
`allow_b_stationary` / `allow_double_buffer_c`）把被枚举和发射的设计点限制为
已有 lowering 支持的那些；关闭的轴不参与打分。本 pass 打开
**A/B-stationary**：被钉住的操作数在移动网格上以**单缓冲**形式常驻
（`k == K`），由 `BuildFullKPipelined` 的 `ForKind::Sequential` 外层循环实现；
若外层流水化，则需要两倍的满 L0 预算。

**dbC=2** 是双累加器 L0C ping-pong，即 tile *i* 的 FIXPIPE drain 与 tile
*i+1* 的 MAD 重叠。它在 `memory_planner=DSA_RP` 与
`memory_planner=PTOAS` 下自动开启。旧版 `PYPTO` planner 仍保留实验性显式
开关（`PassContext(enable_pypto_l0c_double_buffer=True)`，默认关闭），因为
issue #1908 在某些链式 Mat-scratch 布局下仍可能导致操作数缓冲溢出。
`BuildFullKPipelined` 给移动循环加上 `kPipelineDoubleBufferCAttr`，
`CanonicalizeIOOrder` 把两个 store 都浮到两个 matmul 之后
（`matmul, matmul, store, store`），从而让两个累加器生命周期共存。

三种 planner 以不同方式保留该意图。对符合条件的 PTOAS 流水线，
[`LowerPipelineToSlots`](28-lower_pipeline_to_slots.md) 把各 stage 表示成同一分配的
slot；被拒绝的循环继续交给 [`LowerPipelineLoops`](29-lower_pipeline_loops.md)，
PTOAS 自行把对应 stage buffer 放到不同 offset。`PYPTO` 使用
`LowerPipelineLoops` 发出的扁平 depth-2 `pipeline_membership`，由
`MemoryReuse` 的容量门控（#1475）在可负担深度内保持 buffer 分离。`DSA_RP`
也跳过 `MemoryReuse`；它把流水线 stage 分离表示为硬约束，
先运行有界严格搜索，仅当该搜索未找到满足容量的放置时才把流水线意图分离放宽为软
惩罚。dbC=2 要求 full-K，且移动的内层轴至少有两个**完整** tile；固定的外层轴
可以只有一个 tile。发射的循环方向遵循 chooser 的 stationarity/hoist 决策，
peeled 的部分边界不计作 ping-pong stage。因此，以行作为外层时允许 1×2 网格，
以列作为外层时允许 2×1 网格。Mat-scratch（`Acc→Mat`，`tile.assemble`）的
drain 也以相同方式浮动。若 `PassManager` 在一个 planner
下构造却在另一个下运行，会显式报错，因为 pass 列表与 chooser gate 必须一致。
代价模型公式本身与 gate 无关。共存浮动与 `{0, L0C/2}` 不同 offset 的运行时验证
见 [`30-canonicalize_io_order.md`](30-canonicalize_io_order.md)。

上段的 full-K 与移动内层限制只适用于 chooser 发射的 M/N 切分。独立的已有流水线识别器不改变 chooser 的设计空间：它仅在 PyPTO 下，对上文的规范 stationary-panel 模式执行函数级 Acc 保守容量检查后复用相同的双 Acc 机制。

> **这是模型驱动的 tile 选择变更，并非行为中立的重构。** roofline 目标替换了此前以流量最小化为目标的闭式 chooser，因此对 MAD-bound 形状所选的 `(m, n, k)` 与之前不同。代表性形状的前后 tile 在 `test_l0_tile_chooser.py::TestL0TilingRooflineMigration` 中固定下来。

完整的设计依据（带宽 / MAD 数值的 perf-sim 推导、stationarity 与双缓冲的结论）见 chooser 头文件 `l0_tile_chooser.h` 以及 perf-sim 研究文档 `DESIGN_SPACE.md`。`ChooseL0Tile` 的最优解在 `tests/ut/ir/transforms/test_l0_tile_chooser.py` 中由对**同一代价模型**的暴力重新枚举来验证——这是对**求解器**（确认它找到模型的全局最小）的独立检查，而非模型与硬件的对照。

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
            c_acc = pl.tile.matmul_acc(c_iter, sa, sb, ko == 0)
            c = pl.yield_(c_acc)
        # c（即 yield-LHS）持有累加得到的 Acc 类型结果。
        ...
```

### 行窄化的左操作数：种子声明 `compact`

当左操作数的有效行数无法证明等于物理行数时，累加器种子会以*窄化且 compact* 的占位形式生成：

```python
c_l0_init_storage = pl.tile.create([64, 128], pl.INT32, target_memory=Acc, compact=True)
c_l0_init = pl.tile.set_validshape(c_l0_init_storage, 16, 128)
```

`mad` 取 L0A 操作数的**有效**行数作为 M，并以 `ceil(M/16)*16` 的 N-fractal stride 把乘积
写入 L0C（pto-isa `TMatmul.hpp`），只有 compact 的 tile 才会让读取方重新计算该 pitch，
否则一律使用物理行数。`tile.matmul` 的模式来自 `StampCompactForNarrowedAccRows`，而
`tile.matmul_acc` 是**继承**累加器操作数的模式——因此非 compact 的种子会把每一步累加、
以及循环之后的 `tile.store` / `tile.tpush_to_aiv` 一起拖回物理 pitch，令第一个之后的每个
N-fractal 都错位（issue #2470、#2510）。

该模式在 `tile.create` 上**声明**而非事后盖在类型上，是因为只有声明能够存活：
`InferTileMemorySpace` 会对参数发生变化的 call 重新推导类型，从而丢弃 pass 盖上的类型细化，
而 kwarg 每次都会被 deducer 重新读取。生成的契约由 `AccCompactValid` 校验。

### `tile.matmul_acc`

调用方的累加器直接穿过 iter-arg，不生成 `init_cond` 谓词——该累加器在第一次迭代时就已经有效，绝不能被覆写：

```python
for ko, (c_iter,) in pl.pipeline(0, K, k, init_values=(acc_init,), stage=2):
    sa = pl.tile.extract(a_mat, 0, ko, [m, k], target_memory=Left)
    sb = pl.tile.extract(b_mat, ko, 0, [k, n], target_memory=Right)
    c_new = pl.tile.matmul_acc(c_iter, sa, sb)
    c = pl.yield_(c_new)
# c（即 yield-LHS）持有累加得到的 Acc 类型结果。
```

### 带调用方 `init_cond` 的 `tile.matmul_acc`

`pl.tile.matmul_acc(acc_init, a_mat, b_mat, init_cond=user_cond)` 是 split-K 的惯用写法：`user_cond` 标记用户自身归约的第一个 K 步。本 pass 把它与 K-loop 引入的 `ko == 0` 组合，因此只在该步的第一个 L0 block 上覆写累加器：

```python
for ko, (c_iter,) in pl.pipeline(0, k_full, k, init_values=(acc_init,), stage=2):
    sa = pl.tile.extract(a_mat, 0, ko, [m, k], target_memory=Left)
    sb = pl.tile.extract(b_mat, ko, 0, [k, n], target_memory=Right)
    c_new = pl.tile.matmul_acc(c_iter, sa, sb, user_cond and ko == 0)
    c_kmain = pl.yield_(c_new)
# Peeled partial tail (k does not divide K): unpredicated 3-operand form —
# it runs at K offset k_full > 0, so it is never the first block.
sat = pl.tile.extract(a_mat, 0, k_full, [m, k_eff], target_memory=Left)
sbt = pl.tile.extract(b_mat, k_full, 0, [k_eff, n], target_memory=Right)
c = pl.tile.matmul_acc(c_kmain, sat, sbt)
```

字面量 `True` 谓词同样被组合，而不会折回 `tile.matmul`：折回会额外生成一块 L0C buffer，而后端 emitter 本来就会为编译期谓词选出正确指令 —— 参见[算子参考中的 `init_cond`](../ir/05-operators.md)，其中也说明了本 pass 生成的 `ko == 0` 在 `LowerPipelineLoops` 按副本折叠之后 emitter 如何处理。

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
    c_t1_acc = pl.tile.matmul_acc(c_iter, sa, sb, ko == 0)
    c_t1 = pl.yield_(c_t1_acc)
out_t1 = pl.store(c_t1, [256, 0], out_t0)  # 子块 store 到 out[256:512, 0:256]
```

边界子块（当 `m`/`n` 不整除 `M`/`N`）的逻辑尺寸为 `[min(m, M-mi), min(n, N-ni)]` —— 例如 Ascend910B 上的 256×256 FP32 matmul（chooser 选 `m = 192, n = 160`）会切成逻辑尺寸为 `192×160`、`192×96`、`64×160`、`64×96` 的四个子块。对于规范 split-K 改写，每个操作数的物理 Mat shape 会按其有效 boxed layout 粒度向上对齐，而 `valid_shape` 保留逻辑尺寸。该粒度属于 chooser 的容量合法性判断，包括逻辑整块 INT8 N=80 被补齐为物理 N=96 的情况。`tile.matmul` / `tile.matmul_acc` 将相同的物理/有效尺寸区别传播到循环携带的 Acc，`tile.store` 则仍在原逻辑偏移处只传输有效矩形。例如，N 尾块为 16 列的 INT8 Right tile 会以物理 `[K, 32]`、`valid_shape=[K, 16]` 表示，并产生物理 N=32、有效 N=16 的 Acc。

对于规范 split-K 链，同一输出网格包围的是完整的**源 K 归约**，而不是切片最终 Acc。Issue #2232 中，逻辑 INT32 `[16, 1152]` 结果在 Ascend910B 上的物理占用为 `32 × 1152 × 4 = 144 KiB`，因此需要沿 N 切分。每个生成的 N 子块都会运行全部八个源 K block 并 store 结果，然后才开始下一个 N 子块。

### Fits-L0c 链式 matmul（cast-fold）

**Before**（`[128, 128]` 中间值能放进 L0c；`K = 64` 能放进 L0，因此 producer 是单个 matmul）：

```python
c  = pl.tile.matmul(a_mat, b_mat)          # [128, 128] Acc f32 —— 能放进 L0c
cb = pl.tile.cast(c, pl.BF16)              # 若不处理会 lower 成 Vector pto.tcvt
d  = pl.tile.matmul(cb, e_mat)             # 在片上消费 bf16 中间值
out = pl.tile.store(d, [0, 0], out)
```

**After**（cast 被折叠成一次整窗 Acc→Mat assemble；`cb` 的消费者读取 Mat scratch）：

```python
c       = pl.tile.matmul(a_mat, b_mat)                       # 不变（能放进 L0c）
c_mat   = pl.tile.create([128, 128], dtype=pl.BF16, target_memory=Mat)  # L1/Mat scratch
c_mat_t0 = pl.tile.assemble(c_mat, c, [0, 0])                # Acc f32 → Mat bf16（cube pto.tinsert）
d       = pl.tile.matmul(c_mat_t0, e_mat)                    # 在片上读取 scratch
out     = pl.tile.store(d, [0, 0], out)
```

`tile.cast` 被删除。当 producer 需要 K-loop（`k < K`）时，照常发出 K-loop，其 Acc 结果喂给*同一个*单次 `tile.assemble` —— 折叠与 K 切分无关。

## Backend 约束

L0/Mat 容量与 fractal 对齐都来自当前 `BackendHandler`。Pass 优先从 `PassContext::Current()->GetBackendHandler()` 读取，若无活动 context 则回退到 `pypto::backend::GetBackend()->GetHandler()`（例如未包 `PassContext` 直接调用的测试场景）。

| Handler 调用 | 用途 |
| ------------ | ---- |
| `GetL0aCapacityBytes()` | chooser 中 L0a (Left) 容量 |
| `GetL0bCapacityBytes()` | chooser 中 L0b (Right) 容量 |
| `GetL0cCapacityBytes()` | chooser 中 L0c (Acc) 容量 |
| `GetBiasCapacityBytes()` | Bias table 容量；限制 `tile.matmul_bias` 的候选 N |
| `SupportsMatToBiasMove(src, dst)` | 将 Mat-resident bias 物化到 Bias 时的 PTO-ISA dtype 合法性 |
| `GetMatCapacityBytes()` | Mat-scratch gate 中 Mat (L1) 容量 |
| `GetL0FractalAlignment()` | chooser 中 M/N/K 对齐粒度 |
| `GetL0cMAlignment(dtype)` | L0C 容量所用的物理 M 行对齐；Ascend910B INT32 为 32 |
| `GetMinL0TileDim()` | 单轴最小 tile 尺寸 |

因此新增 backend 时，只需要提供这些 handler 接口；本 pass 自身与具体 backend 无关。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

**Properties 声明**：`include/pypto/ir/transforms/pass_properties.h`（`kAutoTileMatmulL0Properties`）

**实现**：`src/ir/transforms/auto_tile_matmul_l0_pass.cpp`

**Chooser 工具**：`src/ir/transforms/utils/l0_tile_chooser.cpp` —— 基于 roofline 代价模型的 L0 tile 选取（在合法对齐网格上穷举；见 [Cost model & design space](#cost-model--design-space-choosel0tile)），未来其它 tiler 也可复用。

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
| 静态 2D、右操作数为 Mat（左为 Mat 或 PV 的 Vec）、输出可放进 L0c 的 `tile.matmul` | 改写为 2 阶段流水化 K-loop（循环体为带谓词的 `tile.matmul_acc` —— 单块 Acc buffer，无 phi）；Vec 左操作数先预存到 Mat |
| 输出超过 L0c、被唯一一个 2D `tile.store` 消费的普通 `tile.matmul`（左右均 Mat） | M/N 切分：`ceil(M/m) × ceil(N/n)` 子块网格，每个子块一个 K-loop 并直接 store 到输出（direct-store） |
| 输出超过 L0c、被**完全作为矩阵乘操作数**消费（链式 matmul）、且 `[M, N]` scratch 能放进 Mat/L1 的普通 `tile.matmul` | M/N 切分到 L1/**Mat** scratch（逐子块 Acc→Mat `tile.assemble`），保留在片上供消费者读取（Mat-scratch） |
| 输出*能放进* L0c、经 `tile.cast(c, bf16/f16)` 降精度、且 cast 结果被**完全作为矩阵乘操作数**消费（链式）的 `tile.matmul` | cast-fold：一次整窗 Acc→Mat `tile.assemble`（cube `pto.tinsert`），并删除 cast —— 无 Vector `pto.tcvt` 往返 |
| 静态 2D、右操作数为 Mat（左为 Mat 或 PV 的 Vec）、输出可放进 L0c 的 `tile.matmul_acc` | 改写为 2 阶段流水化 K-loop（循环体统一为 `matmul_acc`） |
| 同一调用带调用方 `init_cond`（4 操作数） | 同样做 K 切分：循环体携带 `init_cond and ko == 0`，单个直线完整块原样携带 `init_cond`，剥离出的尾块保持 3 操作数 |
| 规范 split-K `create([M,N])` → pipeline（首块 `matmul` + 后续循环携带 `matmul_acc`，或单个谓词化的 `matmul_acc(acc, lhs, rhs, <循环变量> == 0)`）→ 单个 2D store，且物理输出超过 L0c | 在 K-loop 外做 M/N 切分；每个 `[m,n]` 子块完成全部 K 归约后再 store |
| 右（B）操作数为 Vec 的 `tile.matmul[_acc]` | 跳过（B 操作数必须从 L1 送入 L0B） |
| 静态 Mat 矩阵操作数与 `[1,N]` Mat/Bias 源的 `tile.matmul_bias`，输出可放进 L0c 但 K 不可 | K 切分；第一个 block 为 head-peel 出的直线 `matmul_bias`，为每个输出块初始化一次，其余 block 的循环使用 `matmul_acc`（无 `IfStmt`，单一 L0C buffer） |
| 静态 Mat 矩阵操作数与仅使用一次、且到调用之间只有同层 load 的 Mat-resident bias load 的 `tile.matmul_bias`，输出超过 L0c，且只有一次 direct store 或只被后续矩阵乘操作数使用 | 用逐 N 块 tensor→Mat 窗口 load 替换完整 load 后做 M/N 切分；使用与新 `tile.matmul` 相同的 direct-GM 或 Mat-scratch 放置 |
| 左操作数为 Vec，或已位于 Bias 且需要 N 切分的 `tile.matmul_bias` | 跳过；新路径要求原生 Mat 矩阵操作数，且不能发出 Bias-to-Bias 子窗口抽取 |
| 已经是 L0 大小（`(m, n, k) == (M, N, K)`）的 matmul | 不动 |
| 输出超过 L0c 但 M/N 放置不适用——非规范的独立 `matmul_acc`、Vec 左操作数、非矩阵乘操作数消费者、或 `[M, N]` 超过 Mat/L1 的链式 matmul scratch | 以 `PerfHint`（`PH-AT-006`）跳过 |
| `K` 不是 cube 分形 16 的倍数 | 以 `PerfHint`（`PH-AT-007`）跳过——不存在分形对齐的 K 切分 |
| 子字节 dtype | 以 `PerfHint` 跳过 |
| 非 InCore 函数（Orchestration、Opaque） | 不动 |

## Diagnostics

当 pass 决定不改写时，会发出 `PerfHint`（而不是失败）；原 matmul 保持不变并继续走后续流水线。`PerfHint` 编码：

| 编码 | 含义 |
| ---- | ---- |
| `PH-AT-003` | 操作数或累加器使用了子字节 dtype |
| `PH-AT-005` | `ChooseL0Tile` 拒绝了该配置 |
| `PH-AT-006` | 输出超过 L0c，但没有受支持的 M/N 放置。对 `tile.matmul_acc` 而言，这特指位于规范 create/split-K-pipeline/store 链之外、由调用方持有的累加器，或者位于该链之内、但 `init_cond` 不是对循环归纳变量的首块判定的 4 操作数调用。该提示也覆盖 Vec 左操作数、需要 N 子窗口的已在 Bias 中的 `tile.matmul_bias` 源、并非完全作为矩阵乘操作数消费的片上结果、或超过 Mat/L1 容量的链式 matmul scratch。Issue #2232 的规范 split-K 情形不会发出此提示。 |
| `PH-AT-007` | 非 16 对齐的 `K`——不存在分形对齐的 K 切分（任何剥离尾块或整段 K 块的列数都非分形），故该 matmul 保持不变 |
| `PH-AT-008` | `ChooseL0Tile` 返回了 fallback 配置并附带 perf hint |
| `PH-AT-009` | 该 backend 需要 bf16/f16 的片上 Mat scratch（如 Ascend910B），但超大链式 matmul 的中间结果是 f32——在消费 matmul 之前把 matmul 结果 cast 成 bf16/f16；否则留在延后路径上 |
| `PH-AT-010` | fits-L0c 链式 matmul 的 cast 无法折叠进 cube FIXPIPE（FIXPIPE 仅以就近取偶把 `f32 → bf16/f16` 降精度）：源非 f32，或舍入模式不是 `rint`（例如默认的 `round`，或 `floor`/`ceil`/`trunc`/`odd`/`none`）。保留在 Vector `pto.tcvt` 路径——一次 cube→vector→cube 往返，在较大 `[M, N]` 下可能撑爆 Vec buffer。对 f32 结果使用 `mode="rint"` 即可留在 cube 上。 |
| `PH-AT-011` | 带 bias 的 matmul 无法形成合法 Bias 窗口：Mat→Bias dtype 不受 backend 支持、矩阵操作数不在 Mat、N 窗口无法从同 scope 的直接 load 重建、Bias 容量不足，或 M/N/K 不满足 layout box 对齐。该调用保持不变。 |

## 相关 Pass

- [`FlattenTileNdTo2D`](13-flatten_tile_nd_to_2d.md) —— 上游 pass；产生本 pass 所需的静态 2D Mat-resident tile 形状
- [`InferTileMemorySpace`](18-infer_tile_memory_space.md) —— 下游 pass；负责桥接本 pass 故意保留下来的 Vec/Acc 累加器
- [`LowerPipelineLoops`](29-lower_pipeline_loops.md) —— 消费本 pass 产生的 `ForKind::Pipeline` + `pipeline_stages=2`
