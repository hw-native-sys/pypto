# LowerCompositeOps Pass

把组合 (composite) tile / distributed 算子降级 (lower) 为一组基本 tile 算子（`tile.muls`、`tile.adds`、`tile.add`、`tile.sub`、`tile.mul`、`tile.cast`）和分布式原语的组合，使代码生成 (codegen) 不再需要发射高层 (high-level) 指令。当前支持 `tile.sin` / `tile.cos`（FP32 Cody-Waite + Horner）以及 `pld.tensor.*` 分布式集合通信算子（`allreduce`（mesh 与 ring）、`allgather`、`reduce_scatter`、`broadcast`、`barrier`）。mesh allreduce 还可能创建 shape-only 的 `tensor.view`，让 tile load/remote/store 操作一个 2D 展平目标窗口。新的组合算子只需在 Pass 文件内部的分发表 (dispatch table) 里加一条降级规则，无需改动分发器本身。

## 概览 (Overview)

`LowerCompositeOps` 是函数级 (function-level) Pass，对每条 `var = Call(...)` 形式的 `AssignStmt`，若其被调对象出现在 Pass 的降级分发表里，则将其改写为一个 `SeqStmts`。对 `tile.sin` / `tile.cos`，规则会发射固定形态的基本 tile 算子序列：`tile.muls`、`tile.adds`、`tile.add`、`tile.sub`、`tile.mul`、`tile.cast`，先做 Cody-Waite 区间归约 (range reduction，π 拆成 4 段)，再做 9 次奇多项式 Horner 求值。对 `pld.tensor.*` 分布式集合通信算子，规则会发射下文记录的跨 rank recipe；`pld.tensor.allreduce` 在 InCore/composite lowering 中仍保持显式 signal 形态。原始目标 `Var` 仍是最终 `AssignStmt` 的 LHS，因此下游对该名字/身份的引用都保持不变。

host-orchestrator 中的 `pld.tensor.allreduce` 调用会跳过本 Pass：`SynthesizeAllReduceSignals` 先把可选 signal 的 host 调用规范化为显式 signal 形态，`MaterializeCommDomainScopes` 再把 data 和 signal window 放入 comm domain，随后由 `LowerHostTensorCollectives` 降级为内部 builtin dispatch。

该 Pass **仅支持 FP32**。非 FP32 输入会在算子构造时被共享的 `DeduceTileFP32OnlyType` 类型推导器 (deducer) 拒绝（见 `src/ir/op/tile_ops/unary.cpp:94`），因此本降级 Pass 看到的总是良类型的 FP32 操作数，不需要在 dtype 上失败。

对不含已注册组合调用（例如 `tile.sin`、`tile.cos`、`pld.tensor.*` 分布式集合通信算子）的程序，Pass 是**结构性 no-op**：所有其他语句都直接走 `IRMutator::VisitStmt_`。展开生成的只包含基本 tile 算子（`tile.muls`、`tile.adds`、`tile.add`、`tile.sub`、`tile.mul`、`tile.cast`）和分布式原语，mutator 不会再改写它们，因此 Pass 也是**幂等的 (idempotent)**。

**所需 (Requires)**：无。

**产生 (Produces)**：无。

**失效 (Invalidates)**：无。

空的 `PassProperties` 契约（`include/pypto/ir/transforms/pass_properties.h` 中的 `kLowerCompositeOpsProperties`）反映了这一事实：本 Pass 的降级在已有 tile/distributed 词汇以及用于暴露规范展平窗口的 shape-only `tensor.view` 内进行，既不建立任何 `IRProperty`，也不破坏任何 `IRProperty`。

## 运行时机 (When It Runs)

`LowerCompositeOps` 是 `Default` 流水线 `tile_pto_passes` 的**第一个 Pass**（见 `python/pypto/ir/pass_manager.py`），紧跟 `ConvertTensorToTileOps`（位置 12）和 `OptimizeOrchTensors`（位置 13）之后。此时所有 tensor 级三角调用 (`tensor.sin`、`tensor.cos`) 已经被转换注册表 (conversion registry) 改写成 tile 等价物 (`tile.sin`、`tile.cos`)，tile 流水线即将开始 tile-shape 规范化 (canonicalisation)。在 `FlattenTileNdTo2D` 之前完成三角函数降级，可以让本 Pass 与 2D 展平规则解耦——展开生成的所有基本 tile 算子（`tile.muls`、`tile.adds`、`tile.add`、`tile.sub`、`tile.mul`、`tile.cast`）在任意 rank 下都有定义良好的语义。

## 架构 (Architecture)

本 Pass 是单个翻译单元 (translation unit)，即 `src/ir/transforms/lower_composite_ops_pass.cpp`：

```text
src/ir/transforms/lower_composite_ops_pass.cpp
  LoweringBuilder           — 单次调用的暂存区 (Bind + 基本 tile 算子构造器：
                              tile.muls、tile.adds、tile.add、tile.sub、tile.mul、tile.cast
                              + 结构化控制流：EmitFor / EmitForReduce
                              / EmitIf / EmitIfExpr + NotEq 标量比较)
  CompositeLoweringFn       — (call, visited_args, builder) -> 结果表达式
  Lower<Op>Rule             — 每个组合算子一个规则函数（LowerSinRule、
                              LowerCosRule、LowerTensorAllReduceRule ...）
  LookupCompositeRule       — 文件内的「算子名 → 规则」分发表 (kRules)
  LowerCompositeOpsMutator  — 遍历函数，对每个 Call 查表
```

新增一个组合算子的步骤（改动都留在 `lower_composite_ops_pass.cpp` 内）：

1. 写一个 `Lower<Op>Rule(call, args, builder)` 函数。它接收原始 `CallPtr`（按需用 `call->span_`、`call->kwargs_`、`call->op_->name_`）、已 visit 过的参数表达式（已应用 var-remap）以及一个 `LoweringBuilder`，其 `Bind` 助手会为每个中间临时变量追加一条 `AssignStmt`。需要控制流的规则可以用 `builder.EmitFor` / `builder.EmitForReduce` / `builder.EmitIf` / `builder.EmitIfExpr`——每个都接收一个 body 回调，回调里收到的嵌套 builder 与外层共享同一个 temp 计数器，因此发射的临时变量名跨任意嵌套深度都唯一。`LowerTensorAllReduceRule` 是含控制流规则的范例（mesh 使用 ready 屏障，加分块 remote_load+accumulate / 屏障 / store；`LowerTensorRingAllReduceRule` 则通过 `mode` kwarg 分发，增加分块 RS+AG ring 调度）。
2. 在 `LookupCompositeRule` 的 `kRules` 里加一条 `{"<op>", &Lower<Op>Rule}`。

无需修改 mutator。当分发表条目增多——或某条规则需要独立的翻译单元时——再把它拆回 `src/ir/transforms/composite_ops/` 下的独立注册表。

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

连跑两次 `LowerCompositeOps` 会得到与第一次完全相同的 IR：tile recipes 展开后只剩 `tile.muls`、`tile.adds`、`tile.add`、`tile.sub`、`tile.mul`、`tile.cast`，distributed recipes 展开后只剩下文列出的分布式原语。mutator 只改写已注册的组合调用（`tile.sin`、`tile.cos`、`pld.tensor.*` 分布式集合通信算子等），所以第二次访问 body 时不会有任何变化。`tests/ut/ir/transforms/test_lower_composite_ops.py` 中的 sin/cos 与分布式集合通信幂等性测试验证了这一性质。

## `pld.tensor.*` 分布式集合通信算子

本 Pass 同时降级 `pld.tensor.*` 系列的窗口绑定 (window-bound) 分布式集合通信算子。每个集合通信算子都是一个组合 `Call`，展开为 notify / wait + 数据搬运序列。数据搬运原语因算子而异：`allgather` 使用 `pld.tile.put`（基于 TPUT 的推送，经 VEC staging tile 自动分块），`broadcast` 用 `pld.tile.get` 搬运窗口数据（GM→GM 拷贝），`allreduce` 与 `reduce_scatter` 用 `pld.tile.remote_load` 把 peer chunk 拉进 UB tile 并用 `tile.add` 累加。这些规则共享同一套 signal buffer 约定：使用窗口绑定的 INT32 `signal` 矩阵作为跨卡屏障，且**每次调用都需要新分配的 buffer**。

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

任何在降级后仍为符号表达式的目标范围或 partial-valid 范围，都必须在 kernel 中
通过标量参数、循环变量或物理 Tensor shape 参数获得运行时绑定；仅出现在类型元数据
中的符号会在 PTO codegen 阶段被拒绝。完全动态的物理目标维度由该 Tensor 参数绑定。

allreduce 规则先在共享 `signal` cell 上执行 ready 屏障：Phase 2a `AtomicAdd 1` +
Phase 2b `wait ≥1`。之后对完全有效的目标按 UB 大小逐块处理；每个 chunk 完成 peer 归约后执行
`AtomicAdd 1`，等待 `linear_chunk_id + 2`，再写回结果。这个逐块屏障可以阻止
快 rank 覆盖慢 rank 尚未 remote-load 的数据。调用返回时，每个非 self 行停在
`1 + chunk_count`；partial-valid 单矩形路径则停在 `2`。被跳过的 self 行始终保持为 `0`。

**signal buffer 不能跨多次 allreduce 复用**。任意被 wait 的非 self 行上残留的正计数都会让
下次调用的 Phase 2b `wait ≥1` 立刻在旧值上放行，屏障作废，下次读取与上一次
写回直接竞态。需要连续多次 allreduce 的调用者必须为每次调用各自分配新的
signal buffer（`alloc_window_buffer` + `window`）。用户侧的 DSL docstring
（`python/pypto/language/distributed/op/tensor_ops.py::allreduce`）同步标注了这一契约。

所有 wait 谓词都用 `kGe` 而非 `kEq`。单次调用内，每个被 wait 的 cell 都单调递增，因此慢 rank 首次轮询时，快 peer 可能已经把 cell 推过本次 wait 的期望值。此时相等判断会死锁，大于等于判断则不会。

第一版仅支持 `ReduceOp::kSum`；C++ deducer 会拒绝 `Max` / `Min` / `Prod`。

### `pld.tensor.allgather`

签名：`allgather(local_data, target, signal)`。`local_data` 是本 rank 的 chunk（`Tensor` 或 `Tile` `[1, SIZE]`），`target` 是窗口绑定的 `DistributedTensor[NR, SIZE]` 暂存区同时又是结果，`signal` 是 INT32 屏障。基于推送 (push-based) 的展开：

- ``tile.create([1, SIZE], dtype=..., target_memory=Vec)`` — 分配一个 VEC staging tile 供 ``pld.tile.put`` 自动分块使用。``pld.tile.put`` 直接从 ``local_data`` Tensor（或 Tile）源读取 — 不发射显式的 ``tile.load``。
- Phase 1：对 `peer` 从 `0` 到 `NR-1`，`pld.tile.put(target, peer, local_data, put_stage, [my_rank, 0], [0, 0], [1, SIZE])` — 将本 rank 的 chunk 推送到每个 peer 窗口的第 `my_rank` 行。自推送 (`peer == my_rank`) 通过 HCCL 恒等映射实现。`pld.tile.put` 在 SIZE 超过 staging tile 容量时自动分块
- Phase 2a：notify-all（`Set 1`）
- Phase 2b：wait-all（`Ge 1`）
- 返回 `target` — 窗口本身就是汇聚后的 `[NR, SIZE]` 结果（窗口即结果，`DistributedTensor`）

与原始基于拉取 (pull-based) 的 allgather（4 参数带独立 `out` 张量）相比，该推送版本去掉了 `out` 参数和每 peer 的 `pld.tile.get` 汇聚循环。总 HBM 从 `(NR+1)×SIZE` 降至 `NR×SIZE`，代价是窗口在调用方消费结果之前一直处于占用状态。

### `pld.tensor.reduce_scatter`

展开为与 `allreduce` 相同的 5 阶段序列：

- Phase 2a：notify-all（`Set 1`）
- Phase 2b：wait-all（`Ge 1`）
- Phase 3：对每个 peer `p`，`remote_load` 该 peer 的 chunk `r` 并用 `tile.add` 累加到本地 scratch
- Phase 3.5a：re-notify（`AtomicAdd 1`）
- Phase 3.5b：re-wait（`Ge 2`）
- Phase 4：`tile.store` 把归约后的 chunk `r` 写回 `target[r, 0:SIZE]`

`target` 形状为 `[NR, SIZE]`；每个 rank 在调用前暂存全部 `NR` 个 chunk。调用后 rank `r` 的行 `[r, 0:SIZE]` 持有所有 rank 上 chunk `r` 的逐元素和。post-reduce 屏障与 `allreduce` 出于同样的 WAR 原因而必需。

第一版仅支持 `ReduceOp::kSum`；C++ deducer 会拒绝 `Max` / `Min` / `Prod`。

### `pld.tensor.broadcast`

展开为 3 阶段序列：

- Phase 2a：notify-all（`Set 1`）
- Phase 2b：wait-all（`Ge 1`）
- Phase 3：每个 rank 都发射 `tile.create`（VEC staging tile）+ `pld.tile.get(target, peer=root, target, stage)`，把 root 的切片读进自己的 `target`。`peer == root` 时 HCCL 恒等映射让该 get 成为本地空操作，因此 root 保留自己的数据，非 root rank 收到 root 的数据

`root` 是编译时已知的静态 `int` kwarg。

### `pld.tensor.barrier`

纯同步，无数据搬运。展开为 2 阶段序列：

- Phase 2a：notify-all（`Set 1`）
- Phase 2b：wait-all（`Ge 1`）

返回表达式就是同一个 `signal` 张量，支持 `signal = pld.tensor.barrier(signal)` 的 rebind 写法。

### Signal buffer 约定

所有分布式规则在 wait 谓词上都使用 `kGe` 而非 `kEq`。单次调用内 cell 单调递增，但慢的 rank 第一次轮询时，如果快的 peer 已经完成 Phase 3 的数据搬运并开始下一轮 notify，cell 可能已经超过阈值。此时 `kEq` 会死锁，`kGe` 不会。可自重置的写法（调用结束时 set-to-zero / `Eq 0`）受 PTOAS issue #797 阻塞，后续运行时修复落地后才能切换。

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
