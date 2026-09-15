# LowerCompositeOps Pass

把组合 (composite) tile / distributed 算子降级 (lower) 为基础操作，使代码生成 (codegen) 不再需要发射其高层形式。当前支持 `tile.sin` / `tile.cos`（FP32 Cody-Waite + Horner）、`tile.select`（packed predicate mask + TSEL / TSELS）、packed `tile.tquant_mx`，以及 `pld.tensor.*` 分布式集合通信算子（`allreduce`（mesh 与 ring）、`allgather`、`reduce_scatter`、`broadcast`、`barrier`、`all_to_all`、`all_to_all_v`）。mesh 和 ring allreduce 还可能创建保留元数据的 `tensor.view`，让 tile load/remote/store 操作一个 2D 展平目标窗口。

## 概览 (Overview)

`LowerCompositeOps` 是函数级 (function-level) Pass，对每条 `var = Call(...)` 形式的 `AssignStmt`，若其被调对象出现在 Pass 的降级分发表里，则将其改写为一个 `SeqStmts`。对 `tile.sin` / `tile.cos`，规则会发射固定形态的基本 tile 算子序列。`tile.tquant_mx` 会变为值返回的 `tile.tquant_mx_raw` + `tile.tmov_x2zz`（gather_compare 形态 SSA）及显式 workspace tile；公开 FP8 alias 经 `reinterpret_view` / `transpose_view` 补齐。对 `pld.tensor.*` 分布式集合通信算子，规则会发射下文记录的跨 rank recipe。

host-orchestrator 中的 `pld.tensor.allreduce` 调用会跳过本 Pass：`SynthesizeAllReduceSignals` 先把可选 signal 的 host 调用规范化为显式 signal 形态，`MaterializeCommDomainScopes` 再把 data 和 signal window 放入 comm domain，随后由 `LowerHostTensorCollectives` 降级为内部 builtin dispatch。

`tile.sin` / `tile.cos` 规则**仅支持 FP32**。非 FP32 三角函数输入会在算子构造时被共享的 `DeduceTileFP32OnlyType` 类型推导器 (deducer) 拒绝（见 `src/ir/op/tile_ops/unary.cpp:94`），因此这些规则只会看到良类型的 FP32 操作数。分布式规则各自有独立的 dtype 约束；allreduce 如下文所述支持 FP16 和 FP32。

对不含已注册组合调用（例如 `tile.sin`、`tile.cos`、`tile.tquant_mx` 或 `pld.tensor.*` 分布式集合通信算子）的程序，Pass 是**结构性 no-op**：所有其他语句都直接走 `IRMutator::VisitStmt_`。展开生成的只包含基本 tile 算子、内部值返回的 `tile.tquant_mx_raw` / `tile.tmov_x2zz` 形式和分布式原语，mutator 不会再改写它们，因此 Pass 也是**幂等的 (idempotent)**。

**所需 (Requires)**：无。

**产生 (Produces)**：无。

**失效 (Invalidates)**：无。

空的 `PassProperties` 契约（`include/pypto/ir/transforms/pass_properties.h` 中的 `kLowerCompositeOpsProperties`）反映了这一事实：本 Pass 的降级在已有 tile/distributed 词汇以及用于暴露规范展平窗口的 metadata-only `tensor.view` 内进行；partial-prefix ring view 还会携带展平后的 `valid_shape`。本 Pass 既不建立任何 `IRProperty`，也不破坏任何 `IRProperty`。

## 运行时机 (When It Runs)

`LowerCompositeOps` 是 `Default` 流水线 `tile_pto_passes` 的**第一个 Pass**（见 `python/pypto/ir/pass_manager.py`），紧跟 `ConvertTensorToTileOps`（位置 12）和 `OptimizeOrchTensors`（位置 13）之后。此时所有 tensor 级三角调用 (`tensor.sin`、`tensor.cos`) 已经被转换注册表 (conversion registry) 改写成 tile 等价物 (`tile.sin`、`tile.cos`)，tile 流水线即将开始 tile-shape 规范化 (canonicalisation)。在 `FlattenTileNdTo2D` 之前完成三角函数降级，可以让本 Pass 与 2D 展平规则解耦——展开生成的所有基本 tile 算子（`tile.muls`、`tile.adds`、`tile.add`、`tile.sub`、`tile.mul`、`tile.cast`）在任意 rank 下都有定义良好的语义。Packed `tile.tquant_mx` 也在此降级（`FlattenCallExpr` 已先稳定 tuple consumer）。

**与 memory space 的顺序关系：** `LowerCompositeOps` 运行在 `InferTileMemorySpace` **之前**。因此 `tile.tquant_mx` 规则创建的 scratch tile 会在 `tile.create` 上显式盖上 `MemorySpace::Vec`，以便后续 memory planning 仍能拿到地址；这是刻意设计，并不表示 Infer 应该已经跑过。

## 架构 (Architecture)

本 Pass 是单个翻译单元 (translation unit)，即 `src/ir/transforms/lower_composite_ops_pass.cpp`：

```text
src/ir/transforms/lower_composite_ops_pass.cpp
  LoweringBuilder           — 单次调用的暂存区 (Bind + 基本 tile 算子构造器：
                              tile.muls、tile.adds、tile.add、tile.sub、tile.mul、
                              tile.maximum、tile.minimum、tile.cast
                              + 结构化控制流：EmitFor / EmitForReduce
                              / EmitIf / EmitIfExpr + NotEq 标量比较)
  CompositeLoweringFn       — (call, visited_args, builder) -> 结果表达式
  Lower<Op>Rule             — 每个组合算子一个规则函数（LowerSinRule、
                              LowerCosRule、LowerTensorAllReduceRule ...）
  LookupCompositeRule       — 文件内的「算子名 → 规则」分发表 (kRules)
  LowerCompositeOpsMutator  — 遍历函数，对每个 Call 查表
```

新增一个单结果组合算子的步骤（改动都留在 `lower_composite_ops_pass.cpp` 内）：

1. 写一个 `Lower<Op>Rule(call, args, builder)` 函数。它接收原始 `CallPtr`（按需用 `call->span_`、`call->kwargs_`、`call->op_->name_`）、已 visit 过的参数表达式（已应用 var-remap）以及一个 `LoweringBuilder`，其 `Bind` 助手会为每个中间临时变量追加一条 `AssignStmt`。需要控制流的规则可以用 `builder.EmitFor` / `builder.EmitForReduce` / `builder.EmitIf` / `builder.EmitIfExpr`——每个都接收一个 body 回调，回调里收到的嵌套 builder 与外层共享同一个 temp 计数器，因此发射的临时变量名跨任意嵌套深度都唯一。`LowerTensorAllReduceRule` 是含控制流规则的范例（mesh 使用 ready 屏障，加分块 remote_load+accumulate / 屏障 / store；`LowerTensorRingAllReduceRule` 则通过 `mode` kwarg 分发，增加分块 RS+AG ring 调度）。
2. 在 `LookupCompositeRule` 的 `kRules` 里加一条 `{"<op>", &Lower<Op>Rule}`。

多结果规则返回 `MakeTuple`；mutator 会同时映射原始结果及该 tuple 的普通 SSA alias，因此直接投影和 alias 链投影都会暴露同一组 destination。当分发表条目增多——或某条规则需要独立的翻译单元时——再把它拆回 `src/ir/transforms/composite_ops/` 下的独立注册表。

## 算法（`tile.tquant_mx` 规则）

降级创建 dtype 与源一致的 `max` / `scaling` 只写 workspace tile，再 `Bind` 值返回的 `tile.tquant_mx_raw(src, max, scaling)`，用 `TupleGetItem` 投影出 `TupleType{INT8 dst, UINT8 exp}`（与 `tile.gather_compare` 相同的 SSA 形态）。codegen 经 `ResolveTupleResultElements` 解析投影并发射带四个 outs 的 `pto.tquant.mx`。公开 `group_axis` 对齐 PTOAS `grpAxis`：axis1 保持 `[M,K]` 并返回 scale `[M,K/32]`；axis0 先把 `[N,K]` 转置为 `[K,N]` 再返回 `[K/32,N]`。两种形式再 `Bind` 值返回的 `tile.tmov_x2zz(exp, tmp)`。Axis1 tmp 容量为 `64 + ceil(rows/16)*cols` 字节（通常 32 字节对齐）；axis0 使用 `TMovDnTo2Zz` 所需的最小 32 字节 Vec pad。Axis0 遵循 pto-isa `TMovDnTo2Zz`（pin `be5ccb76`）：DN `[M̂,N]` → ZZ `[N,M̂]` row/row，再经零拷贝 `tile.transpose_view` 得到公开 `[M̂,N]` col/col scale。最后补上零拷贝 FP8 data alias 与 FP8E8M0 scale alias。workspace 参数上的 Write effect 保证公开结果未消费时 Call 也不会被 DCE 删掉。

在 Ascend950 上，公开的 quant data 与 scale 可以在同一个 InCore mixed task 内直接供 `matmul_mx` 使用；[ExpandMixedKernel](24-expand_mixed_kernel.md#mx-scale-的-v2c-传输) 会把两个结果直接经 V2C 传递。

mutator 把本 Pass 产出的 `MakeTuple`（及其 SSA alias）记入私有 `composite_tuples_`，再折叠 `TupleGetItem`，避免滥用全局 `var_remap_` 去 inline 任意 `v = (a, b)`。内部 `tile.tquant_mx_raw` / `tile.tmov_x2zz` 不注册为组合规则，Pass 仍然幂等。

## 算法（`tile.select` 规则）

`tile.select(cond, on_true, on_false)` 是免 scratch 的选择接口；`tile.sel` / `tile.sels` 仍是与 PTO 指令 1:1 对应、由调用方自行管理 scratch 的形式。

**条件恒为 packed predicate mask** —— 即针对*当前*结果几何的 `tile.cmp` / `tile.cmps` 结果类型：`[M, roundup(ceil(N/8), 32)] UINT8`，`valid_shape[1] = ceil(valid_N/8)`。`IsPackedPredicateMask`（`include/pypto/ir/type_inference.h`）是唯一的判定点，由 `DeduceTileSelectType` 与本规则共用，因此掩码到达时对 TSEL 和 TSELS 都已就绪，规则完全不再合成比较。

普通的 0/1 值 tile **刻意不作为**第二种可接受形式。掩码的 valid 范围计的是承载**字节**，而值的 valid 范围计的是**元素** —— 两者单位不同却可能数值巧合：全有效源上的 `[M, 32]` UINT8 掩码 valid 为 `[M, 4]`，与一个只有 4 列有效的 `[M, 32]` 值 tile 完全一样。因此任何基于几何的分类器都存在猜错的情形，一旦猜错就会对承载字节重新比较，悄悄丢弃其中打包的谓词位。持有值的调用方请用 `tile.cmps(value, 0, cmp_type=1)` 显式转换。

**按两个值分支选择 PTO 形式：**

| `on_true` | `on_false` | 发射 | 原因 |
| --------- | ---------- | ---- | ---- |
| tile | tile | `tile.create`(TSEL scratch) + `tile.sel(mask, T, F, tmp)` | TSEL 直接接受两个 tile 源，无需物化任何东西 |
| tile | 标量 | `tile.create`(TSELS scratch) + `tile.sels(mask, T, tmp, F)` | 正好是 TSELS 的形态 `dst = mask ? src : scalar`，一条指令 |
| 标量 | tile | `tile.full(T)` + `tile.create`(TSEL scratch) + `tile.sel(mask, full, F, tmp)` | TSELS 没有"标量在真值侧"的形式，只能物化标量后走 TSEL |

有两种情况会把 tile/标量那一行也推到最后这条路径上，因为目标架构没有对应的 TSELS 形式：**A2/A3 的 8 位整数**（`BackendHandler::SupportsTselsDataType`，与 `pto.tsels` 发射器检查的是同一个谓词，因此 pass 与发射器不会走偏），以及 dtype 无法与源配对的标量。在 pass 里判掉这一点，才能保证程序可编译 —— 发射 `tile.sels` 只会在很晚的发射器 dtype 检查处失败。

`tile.full` 是 tile 流水线唯一能降级的标量广播：`tile.expands` 没有 PTO 发射器，`pl.expands` 自身就过不了 codegen。这就把标量分支限制为**字面量**、把结果限制为**静态** extent；两者都由 `DeduceTileSelectType` 校验，因此诊断落在用户的调用点上，而不是以 `tile.full` 报错的形式从本 Pass 内部冒出来。

`tile.full` 还会把结果标成全部有效，而 `tile.sel` 从它的 `lhs` 读取结果的 valid 范围，所以被物化的分支会把尾部收窄的 select 重新放宽。规则会用另一分支的 view 给新建的调用重新定型 —— 只覆盖 view，这样改写后的调用仍能通过 print → parse 往返。

**scratch 形状由后端拥有。** 由 `BackendHandler::GetTselScratchSpec` / `GetTselsScratchSpec` 提供：A2/A3 的 TSEL 用 `UINT32 [1, 16]`，TSELS 则按一整行完整物理源、以源 dtype 计算（`src_dtype [1, N]`）；A5 两者都用 `UINT8 [1, 32]`。与 `tile.tquant_mx` 规则一样，scratch 的 `tile.create` 显式标注 `MemorySpace::Vec`，因为本 Pass 在 `InferTileMemorySpace` 之前运行。

**操作数必须是 rank 2。** `tile.sels` 只接受 rank 2，而本可以折叠 ND 操作数的 `FlattenTileNdTo2D` 运行在本 Pass **之后**（slot 14 vs 13），且并不覆盖 cmp/sel 家族 —— 因此 ND 的 select 要么在本规则里失败，要么晚些在 `TileOps2D` 校验器处失败。`DeduceTileSelectType` 提前拒绝并提示改用 `tile.reshape`。

**两个 tile 分支必须在物理 shape、valid 范围和 dtype 上都一致。** TSEL 逐元素读取两个源、不做广播，且 `src0` / `src1` / `dst` 只有一种元素类型。接受 `[1, N]` 对 `[M, N]` 等于承诺指令给不出的语义 —— 而且结果会取决于窄的那个操作数在哪一侧：shape 会广播成 `[M, N]`，valid 范围却来自被查询的那个分支。把 FP16/FP32 提升成 FP32 结果则会发射一条 ptoas 拒绝的 TSEL：`expects src0, src1, and dst to have the same element type`。因此 `tile.select` 在类型推导阶段就拒绝这三类不一致，并提示改用显式的广播与 cast 算子。

## 算法 (Algorithm，sin / cos 规则)

`src/ir/transforms/lower_composite_ops_pass.cpp` 中的 `LowerSinCos` 由 `is_cos` 参数化。mutator 重写的是 `VisitStmt_(const AssignStmtPtr&)`，而不是 `VisitCall`，因为每个三角算子要展开成 ~33 条语句，每条都需要新临时 `Var`。在语句级 (statement level) 工作让规则可以通过 builder 直接把语句追加到外围序列里。

### 区间归约 (Range Reduction，4 段 π Cody-Waite)

目标是把 `x` 写成 `x = k·π + t`（sin）或 `x = k·π + π/2 + t`（cos），其中 `t ∈ [-π/2, π/2]` 而 `k` 是整数。FP32 不能精确表示 π，所以单步 `x - k·π_fp32` 每次乘法引入约 1e-7 的相对误差，区间归约误差会随 `|k|` 线性放大。Cody-Waite 把 π 拆成一个快速取整的 head 加上若干（这里是 4 段）小修正，使消去 (cancellation) 误差只在最细尺度上才丢失精度：

```text
π ≈ PI_V2 + PI_C1 + PI_C2 + PI_C3 + PI_C4
```

`t` 通过链式减法计算，每段消耗一个修正：

```text
t0 = x  - k_f * PI_V2
t1 = t0 - k_f * PI_C1
t2 = t1 - k_f * PI_C2
t3 = t2 - k_f * PI_C3
t4 = t3 - k_f * PI_C4
```

对于 **sin**，`k_f = float(round(x · PI_INV))`，即 `tile.cast` 取 `ROUND` 模式（最近偶数远离零）。对于 **cos**，取整再叠加 `0.5` 偏移，使 `k` 表示中点最接近 `x` 的 `π` 倍数：

```text
k_f = float(rint(x · PI_INV + 0.5))   ; mode RINT (round-half-to-even)
```

cos 路径还在归约中段加上 `π/2`，并将其同样按 Cody-Waite 拆成 `PI_HALF_HEAD + PI_HALF_TAIL`：`PI_HALF_HEAD` 折叠到 `PI_C1` 与 `PI_C2` 之间，`PI_HALF_TAIL` 在 `PI_C4` 之后追加，保证每次加减都与周围在同一量级，把灾难性消去 (catastrophic cancellation) 区间分摊到 5+2 段修正上。

### 符号计算 (Sign Computation)

`k` 求出之后，可以无条件地用浮点算术算出 `sign`：

```text
sign = floor(k_f / 2) · 4 + k_f · (-2) + 1
     = (-1)^k
```

恒等式 `floor(k/2)·4 - 2·k + 1` 对偶数 `k` 给 `+1`，对奇数 `k` 给 `-1`。证明把 `k = 2m + r`、`r ∈ {0, 1}` 代入即可：

```text
floor(k/2) = m
floor(k/2)·4 - 2·k + 1 = 4m - 2(2m + r) + 1 = 1 - 2r
```

`r = 0` 时为 `+1`，`r = 1` 时为 `-1`。Pass 用 6 步实现：

```text
half_k     = k_f * 0.5
floor_hk_i = int32(floor(half_k))         ; tile.cast mode FLOOR
floor_hk_f = float(floor_hk_i)
floor_x4   = floor_hk_f * 4.0
neg2_k     = k_f * (-2.0)
sign_pre   = floor_x4 + neg2_k
sign       = sign_pre + 1.0
```

### Horner 多项式 (Horner Polynomial)

`t ∈ [-π/2, π/2]` 上 `sin(t)` 用 9 次奇多项式 `t · P(t²)` 近似，其中：

```text
P(u) = (((R0·u + R1)·u + R2)·u + R3)·u + 1
```

`P(u)` 末尾的常数 `1` 对应 Taylor 级数的 `t¹` 项，`R3 ≈ -1/6`、`R2 ≈ 1/120`、`R1 ≈ -1/5040`、`R0 ≈ 1/362880` 对应高阶奇次项，并按 `[-π/2, π/2]` 上的 minimax 精度做了微调。实现：

```text
t2     = t * t
p_r0   = t2 * R0
p_r1   = p_r0 + R1
p_t2_r1= p_r1 * t2
p_r2   = p_t2_r1 + R2
p_t2_r2= p_r2 * t2
p_r3   = p_t2_r2 + R3
p_t2_r3= p_r3 * t2
p_one  = p_t2_r3 + 1.0
t_p    = t * p_one
out    = sign * t_p
```

sin 与 cos 共用同一组多项式系数：cos 路径只在区间归约阶段不同，多项式入口处 `t` 已经位于 `[-π/2, π/2]`，无需另一组系数。

### sin 与 cos 对照 (Sin vs Cos at a Glance)

| 步骤 | sin | cos |
| ---- | --- | --- |
| 1. k 取整 | `round(x · 1/π)`（mode `ROUND`） | `rint(x · 1/π + 0.5)`（mode `RINT`） |
| 2. 区间归约 | `x - k·π`（4 段） | `x - k·π + π/2`（4 段 + 2 段 π/2） |
| 3. 符号 | `(-1)^k` | `(-1)^k`（同恒等式，`k` 不同） |
| 4. Horner | `t · P(t²)` | `t · P(t²)`（同多项式） |
| 5. 结果 | `sign · t · P(t²)` | `sign · t · P(t²)` |

## 常量 (Constants)

所有常量均为 FP32 字面量（即 `src/ir/transforms/lower_composite_ops_pass.cpp` 顶部附近的 `k*` 字面量，与 `gitcode.com/cann/pypto:framework/src/interface/tileop/vector/unary.h` 上游参考实现一致）：

| Symbol | C++ literal | Role |
| ------ | ----------- | ---- |
| `PI_INV` | `0.31830988732818603515625f` | `1/π` (head) |
| `PI_V2` | `3.140625f` | π head (Cody-Waite part 1) |
| `PI_C1` | `0.0009670257568359375f` | π split-1 |
| `PI_C2` | `6.2771141529083251953125e-7f` | π split-2 |
| `PI_C3` | `1.21644916362129151821e-10f` | π split-3 |
| `PI_C4` | `-1.0290623200529979163e-13f` | π split-4 |
| `PI_HALF_HEAD` | `1.57079637050628662109375f` | π/2 head (cos only) |
| `PI_HALF_TAIL` | `-4.371139000189375e-8f` | π/2 tail (cos only) |
| `HALF` | `0.5f` | k-pre offset (cos), sign step |
| `M4` | `4.0f` | sign step |
| `NEG2` | `-2.0f` | sign step |
| `ONE` | `1.0f` | sign + Horner constant term |
| `R0` | `2.604926501e-6f` | Horner coeff (degree 9) |
| `R1` | `-1.980894471e-4f` | Horner coeff (degree 7) |
| `R2` | `8.333049340e-3f` | Horner coeff (degree 5) |
| `R3` | `-1.666665792e-1f` | Horner coeff (degree 3) |

`tile.cast` 取整模式（与 `src/ir/op/tile_ops/unary.cpp` 注册一致）：

| Symbol | Value | Meaning |
| ------ | ----- | ------- |
| `kCastModeNone` | `0` | no rounding (typically int → float) |
| `kCastModeRint` | `1` | round-half-to-even |
| `kCastModeRound` | `2` | round-half-away-from-zero |
| `kCastModeFloor` | `3` | round toward `-∞` |

## 数值性质 (Numerical Properties)

- **绝对误差 (absolute error)**：在 `|x| ≤ 2π · 1024` 范围内 ≤ ~1e-5（由 `tests/ut/ir/transforms/test_lower_composite_ops_numerical.py` 与 NumPy 对照验证）。一个周期内观察到的最大绝对误差约为 1 ulp ≈ 1.19e-7。
- **区间归约失效 (range-reduction breakdown)**：当 `|x| ≈ 2^17` 时，`x` 自身的 FP32 表示已经丢掉小数精度，无论 π 修正项再多，区间归约误差都会主导整体误差。本实现选用的 4 段 Cody-Waite 拆分是 CANN/PyPTO 标准方案，在所有测试 `x` 量级上都与上游参考实现表现一致。
- **dtype**：仅 FP32。FP16、BF16、整型输入会在算子构造时被拒绝（早于本 Pass）——参见 `tests/ut/ir/operators/test_tensor_ops.py`（tensor.sin/cos 拒绝）与 `tests/ut/ir/operators/test_tile_ops.py`（tile.sin/cos 拒绝）的拒绝用例。
- **NaN/Inf**：NaN 输入会传播为 NaN 输出（多项式本身保留 NaN）。Inf 输入会产生不确定值，因为区间归约 `k = round(x/π)` 步会溢出；这与文档约定的 `|x| ≤ 2^17` 有效范围一致。

## 幂等性 (Idempotency)

连跑两次 `LowerCompositeOps` 会得到与第一次完全相同的 IR：recipes 展开后只剩基础 tile 算子、内部 `tile.tquant_mx_raw` / `tile.tmov_x2zz` 以及下文列出的分布式原语。这些结果都不是已注册的组合调用，所以第二次访问 body 时不会有任何变化。

`tile.tquant_mx` 规则发射的是非组合的 `tile.tquant_mx_raw` 与 `tile.tmov_x2zz`，因此 MX 降级同样幂等。

## `pld.tensor.*` 分布式集合通信算子

本 Pass 同时降级 `pld.tensor.*` 系列的窗口绑定 (window-bound) 分布式集合通信算子。每个集合通信算子都是一个组合 `Call`，展开为 notify / wait + 数据搬运序列，外加自清理尾声。数据搬运原语因算子而异：`allgather` 与 ring `allreduce` 使用 `pld.tile.put`（基于 TPUT 的推送，经 VEC staging tile 自动分块），`broadcast` 用 `pld.tile.get` 搬运窗口数据（GM→GM 拷贝），mesh `allreduce` 与 `reduce_scatter` 用 `pld.tile.remote_load` 把 peer chunk 拉进 UB tile。allreduce 与 reduce-scatter 都根据规约类型选择 `tile.add`、`tile.maximum`、`tile.minimum` 或 `tile.mul`。七条规则共享同一套**自清理信用屏障协议**（`LoweringBuilder::EmitBarrier` + `EmitEpilogueReset`）—— 参见下方[屏障-信号协议](#屏障-信号协议) —— 因此 `signal` buffer 可以在连续调用之间复用，甚至在 `for` / `while` / `if` 内部也可以。

### 屏障-信号协议

每次调用发出 `N` 个屏障 —— 对每个 peer cell 做 `AtomicAdd(1)`，然后等待
`Wait(>= g)`，其中 `g` 在**仅本次调用内部**向上计数（每次新调用都从 1 重新开始，
由 `LoweringBuilder::EmitBarrier` 的调用局部 `barrier_count_` 实现）。主体之后，
尾声（`LoweringBuilder::EmitEpilogueReset`）把本次调用的总信用 `N` 通过一次
`AtomicAdd(-N)` 从每一个非 self cell 中减回去。由于原子加法与减法可交换，
一旦所有 rank 都完成本次调用的尾声，signal 可证明地恢复为全零 —— 不存在
跨调用状态需要管理，因此*下次*调用的同一个 signal 也从 generation 1 重新开始。

`N` 可以是**运行时常量**（`pld.system.notify` 的 `value` 只需要 `ScalarType`），
因此 mesh allreduce 的每块信用计数不需要是编译期常量 —— 与屏障计数本身不同，
后者始终是小型编译期序列（`1`、`2`……），因为每个 `Wait` 的 `expected`
必须在不知道外围（可能动态的）循环执行多少次的情况下就能解析。

所有 wait 谓词都用 `kGe` 而非 `kEq`：快 peer 可能在慢 rank 轮询前就把
cell 推过本次 wait 的期望值，因此相等判断会死锁。同理，`kSet` 绝对不能与
`kAtomicAdd` 混用在同一个 cell 上 —— set 可能会覆盖已经被推高的计数器。

mesh（`[NR, 1]`，每 rank 一个 cell）和 ring（`[2*(NR-1), NR]`，每轮一行）
signal 使用互不兼容的 cell 寻址方式，因此共享同一个 buffer 会因形状不匹配
被拒绝（`ValidateMeshSignalShape`） —— 这是协议仍然施加的唯一限制。

### `pld.tensor.allreduce`

对于完全有效的 packed 目标，mesh 降级会创建逻辑
`[1, 所有维度乘积]` 视图，并用最大 16 KiB 的物理 tile 遍历。若静态已知的范围
小于预算，块宽会收缩到能够覆盖它的最小 32-byte 对齐物理宽度，既避免小
allreduce 仍预留完整 16-KiB tile，又满足 PTO tile 的对齐要求。尾块通过
`tile.load` 和 `pld.tile.remote_load` 同时携带
`valid_shape=[1, min(chunk, remaining)]`，因此分配保持静态而实际读写范围精确。
如果 ND 目标带有 partial `TensorView.valid_shape`，Pass 会保留可表示的
`[rows, cols]` 矩形，并沿用单矩形路径完成归约。常量有效矩形使用紧凑物理 shape；
符号型有效范围则在源 tensor 的静态物理矩形能放入一个 16-KiB chunk 时回退使用该矩形。
过大的 partial 矩形、strided 目标、DN partial view 和无法按 leading-dimension collapse
表示的 partial 区域会被明确拒绝。

ring 降级在 reduce-scatter 和 allgather 阶段使用同一个 packed 2D 视图，并用
**TPUT 推送**（`pld.tile.put`，非原子）代替远程加载来搬运数据。
完全有效的目标会变为 `[1, SIZE]`；连续 partial prefix 保留物理 shape
`[1, product(target.shape)]`，并携带逻辑
`TensorView.valid_shape=[1, product(target.valid_shape)]`。FP32 保留均衡的
`floor(i * SIZE / NR)` segment 边界；FP16 把每个内部
边界向上对齐到 16 个元素并限制在 `SIZE` 内，因此每个非空 segment 和 UB
subchunk 都从 32 字节对齐地址开始。每个 subchunk 中，各 rank 先把接收 slot 的
**自身值**读入寄存器 tile（该 slot 在左邻居的推送落地前保持稳定 —— 只有自身值），
随后在 ready generation (2k+1) 上做屏障，再把发送 subchunk 通过 TPUT 推送到
**右邻居**的同一下标 slot；push-done 屏障 (2k+2) 之后才本地读取、归约并写回接收 slot。
共享的 VEC staging tile 通过 `tile.set_validshape` 收窄到每次传输的精确
`valid_cols`，推送传输携带该动态范围 —— PTOAS >= v0.55 接受 `tput` 的动态
partition-view 形状（hw-native-sys/PTOAS#1069），因此无需填充窗口即可保留
ragged 与 FP16 尾部。非原子推送 + 本地归约保留了所有 `ReduceOp`
（Sum/Max/Min/Prod）——只有远端原子 `TPUT<AtomicAdd>` 才只支持 Sum。
该方案无需在公开 tensor 布局中插入空洞，也能支持非整除输入和 `SIZE < NR`。
每一轮的每个 subchunk 都使用本调用局部的 ready + push-done generation 对
做屏障；尾声随后把 `2 * chunk_count`（跨轮统一，因为每轮的 subchunk 循环
共享相同边界）从 signal 的每一行中减去。

任何在降级后仍为符号表达式的目标范围或 partial-valid 范围，都必须在 kernel 中
通过标量参数、循环变量或物理 Tensor shape 参数获得运行时绑定；仅出现在类型元数据
中的符号会在 PTO codegen 阶段被拒绝。完全动态的物理目标维度由该 Tensor 参数绑定。

allreduce 规则先在共享 `signal` cell 上执行 ready 屏障（generation 1）。之后对完全
有效的目标按 UB 大小逐块处理；每个 chunk 完成 peer 归约后对本调用局部的
generation（`2`、`3`……）做屏障，再写回结果 —— 阻止快 rank 覆盖慢 rank
尚未 remote-load 的数据。尾声随后把本次调用的总信用
（`1 + chunk_count`，构建为 IR 表达式，因为 `chunk_count` 可能依赖运行时
范围）从每个非 self 行中减去。partial-valid 单矩形路径恰好发出两个屏障
（ready + post-reduce），其尾声减去 `2`。

signal buffer 可以在连续 allreduce 调用之间安全管理复用 —— 包括对一个符号
range 进行 mesh allreduce，其信用总数只是运行时计算表达式，而非编译器必须知道
的值。参见上方[屏障-信号协议](#屏障-信号协议)。

mesh 和 ring 降级均支持 FP16、FP32，以及任意正元素数量下的
`ReduceOp::kSum`、`kMax`、`kMin` 和 `kProd`。

### `pld.tensor.allgather`

签名：`allgather(local_data, target, signal)`。`local_data` 是本 rank 的 chunk（InCore 路径上为普通 `Tensor` `[1, SIZE]`），`target` 是窗口绑定的 `DistributedTensor[NR, SIZE]` 暂存区同时又是结果，`signal` 是 INT32 屏障。基于推送 (push-based) 的展开：

- ``tile.create([1, stage_cols], dtype=..., target_memory=Vec)`` — 分配一个 VEC staging tile 供 ``pld.tile.put`` 自动分块使用。对于静态 extent，``stage_cols`` 先取 ``min(SIZE, chunk_elements)``，在小于完整 chunk 时再向下取整到 32-byte 行边界；例如 FP32 ``SIZE=17`` 得到 ``stage_cols=16``，而不是 17。短于一个对齐单元（FP32 为 8 个元素，packed-FP4 为 64 个元素）的传输，不存在既对齐又不大于传输的正 stage 宽度，因此仍不受支持。符号 extent 使用完整 chunk 上界 — 与 `MakeTputStageShape` / `chunk_cols` 相同的静态 UB 约定，因此运行时宽度 17 仍会分配 4096 个 FP32 元素。几何计算采用物理存储 bit 宽度，因此一个 16-KiB stage 可容纳 4096 个 FP32 或 32768 个 packed-FP4 逻辑元素。该 stage 是有界中转缓冲，**不是**传输的副本：pto-isa 从 partition view 读取完整 extent 并把传输在 stage 上做二维滑动，因此按 ``SIZE`` 来分配 stage 只会浪费 UB（``[1, 65537]`` 的 FP32 stage 为 256 KiB，会超出 VEC 预算）。``pld.tile.put`` 直接从 ``local_data`` Tensor 源读取 — 不发射显式的 ``tile.load``。
- Phase 1：对 `peer` 从 `0` 到 `NR-1`，`pld.tile.put(target, peer, local_data, put_stage, [my_rank, 0], [0, 0], [1, SIZE])` — 将本 rank 的 chunk 推送到每个 peer 窗口的第 `my_rank` 行。自推送 (`peer == my_rank`) 通过 HCCL 恒等映射实现。`pld.tile.put` 在 SIZE 超过 staging tile 容量时自动分块
- Phase 2：屏障（generation 1）+ 尾声（从每个非 self cell 减去 1）
- 返回 `target` — 窗口本身就是汇聚后的 `[NR, SIZE]` 结果（窗口即结果，`DistributedTensor`）

与原始基于拉取 (pull-based) 的 allgather（4 参数带独立 `out` 张量）相比，该推送版本去掉了 `out` 参数和每 peer 的 `pld.tile.get` 汇聚循环。总 HBM 从 `(NR+1)×SIZE` 降至 `NR×SIZE`，代价是窗口在调用方消费结果之前一直处于占用状态。

### `pld.tensor.reduce_scatter`

展开为与 `allreduce` 相同的 5 阶段序列：

- Phase 2a：notify-all（`AtomicAdd 1`）
- Phase 2b：wait-all（`Ge 1`）
- 尾调用：`EmitEpilogueReset`（自清理信用屏障）
- Phase 3：对每个 peer `p`，`remote_load` 该 peer 的 chunk `r` 并用与 `ReduceOp` 匹配的 tile 算子（`tile.add` / `tile.maximum` / `tile.minimum` / `tile.mul`）累加到本地 scratch
- Phase 3.5a：re-notify（`AtomicAdd 1`）
- Phase 3.5b：re-wait（`Ge 2`）
- Phase 4：`tile.store` 把归约后的 chunk `r` 写回 `target[r, 0:SIZE]`

`target` 形状为 `[NR, SIZE]`；每个 rank 在调用前暂存全部 `NR` 个 chunk。调用后 rank `r` 的行 `[r, 0:SIZE]` 持有所有 rank 上 chunk `r` 的逐元素归约结果（按 `ReduceOp` 取 Sum/Max/Min/Prod）。post-reduce 屏障与 `allreduce` 出于同样的 WAR 原因而必需。

四种 `ReduceOp` 均受支持 —— `kSum`、`kMax`、`kMin`、`kProd` —— 通过共享的 `Reduce()` helper 路由，与 allreduce 规则使用的分发一致。

### `pld.tensor.broadcast`

展开为 3 阶段序列：

- Phase 2：屏障（generation 1）
- Phase 3：每个 rank 都发射 `tile.create`（VEC staging tile，与 allgather 一样上限为一个 16-KiB chunk）+ `pld.tile.get(target, peer=root, target, stage)`，把 root 的切片读进自己的 `target`。`peer == root` 时 HCCL 恒等映射让该 get 成为本地空操作，因此 root 保留自己的数据，非 root rank 收到 root 的数据。由于 stage 不再由 target 的 extent 推导，**动态** target shape 现已被接受 — 动态维度直接取 chunk 上限
- 尾声：从每个非 self cell 减去 1

`root` 是编译时已知的静态 `int` kwarg。

### `pld.tensor.all_to_all`

对称推送：每个 rank 通过 `pld.tile.put` 把 `input[dest, :]` 写入每个 peer 窗口的第 `my_rank` 行，然后做屏障。共享 VEC stage 与 allgather 相同，是有界的 `[stage_rows, stage_cols]` 中转缓冲。

### `pld.tensor.all_to_all_v`

可变大小推送（`MPI_Alltoallv` 模式）。每个目的地通过同一个有界 2-D stage 传输 `clamp(send_counts[dest], 0, MAX_RECV)` 行；`target.shape[0] % NR == 0` 仍是负载约束，以便接收方按行 `s * MAX_RECV` 定位发送方 `s`。运行时行数是动态 partition-view 维，stage 本身仍是静态 chunk 上界。

### `pld.tensor.barrier`

纯同步，无数据搬运。展开为屏障（generation 1）加上其尾声（从每个非 self cell 减去 1）。

返回表达式就是同一个 `signal` 张量，支持 `signal = pld.tensor.barrier(signal)` 的 rebind 写法。在自清理协议下，每次调用都从 generation 1 重新开始，因此 rebind 不再需要链到任何先前状态。

### Signal buffer 约定

每次调用发出 `N` 次屏障 — 向每个 peer cell 做 `AtomicAdd(1)`，再 `Wait(>= g)`，其中 `g` **只在该次调用内**递增（每次新调用经 `EmitBarrier` 的调用局部 `barrier_count_` 从 1 重新开始）。主体结束后，尾声（`EmitEpilogueReset`）用一次 `AtomicAdd(-N)` 从每个非 self cell 减去本次调用的总信用 `N`。原子加与减可交换，因此一旦每个 rank 完成尾声，signal 可证明再次全零 — 没有跨调用状态，下一次在同一 signal 上的调用也从 generation 1 开始。

所有分布式规则在 wait 谓词上都使用 `kGe` 而非 `kEq`。快的 peer 可能在慢 rank 第一次轮询前就把 cell 推进到超过阈值，此时 `kEq` 会死锁，`kGe` 不会。同理，`kSet` 绝不能与 `kAtomicAdd` 混用在同一组 cell 上。

## 实现要点 (Implementation Notes)

mutator 重写 `VisitStmt_(const AssignStmtPtr&)` 而不是 `VisitCall`，原因是每个三角算子要往外围序列里塞约 33 条语句。如果在 `VisitCall` 内做拼接，需要让一个表达式返回多个表达式，`IRMutator` 并不支持；改在 `VisitStmt_` 里做，`LowerSinCos` 可以直接构建一个 `vector<StmtPtr>`，并视情况返回单条绑定 `AssignStmt` 或新的 `SeqStmts`。

每个中间结果都绑定到一个用 `auto_name::BuildName` 生成的临时 `Var`，base 名取用户给的目标名。mutator 的 `temp_counter_` 通过每个 `LoweringBuilder` 按引用共享，确保函数内多个三角调用之间临时名不会冲突。

`tile.cast` 模式 `RINT`（cos）、`ROUND`（sin）、`FLOOR`（sign）、`None`（int↔float）来自 tile 算子注册表的枚举（`src/ir/op/tile_ops/unary.cpp`）。模式选择对正确性至关重要：sin 中 `k` 用 `ROUND` 保持以零为中心对称，使 Horner 多项式看到的 `t` 分布均匀；cos 中 `k` 用 `RINT` 与 `+0.5` 偏移配合，确保偶数 `k` 对应 `π/2` 的偶数倍。

## 相关 (Related)

- **Issue**：[#1289 — Add FP32-only `tile.sin` / `tile.cos` and a lowering pass](https://github.com/hw-native-sys/pypto/issues/1289)。
- **参考实现 (reference implementation)**：`gitcode.com/cann/pypto:framework/src/interface/tileop/vector/unary.h` —— 本 Pass 的常量与算子序列与该上游 CANN/PyPTO 实现逐字对应。
- **算子推导器 (op deducer)**：`src/ir/op/tile_ops/unary.cpp:94` 的 `DeduceTileFP32OnlyType` —— 在算子构造时强制 FP32-only。
- **转换注册表 (conversion registry)**：`src/ir/transforms/op_conversion_registry.cpp` 中的 `RegisterSimple("tensor.sin", "tile.sin")` 与 cos 对应项 —— 上游 tensor-to-tile 改写，产出本 Pass 消费的 `tile.sin` / `tile.cos` 调用。
- **测试**：`tests/ut/ir/transforms/test_lower_composite_ops.py`（结构）与 `tests/ut/ir/transforms/test_lower_composite_ops_numerical.py`（NumPy 数值对照）。
- **MX 量化测试**：`tests/ut/codegen/test_quant_mx_codegen.py`（tuple 消费与内存规划）。
