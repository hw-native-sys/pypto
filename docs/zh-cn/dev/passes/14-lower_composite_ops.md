# LowerCompositeOps Pass

把组合 (composite) tile 算子降级 (lower) 为一组基本算术 tile 算子的组合，使代码生成 (codegen) 不再需要发射高层 (high-level) 指令。当前只支持 `tile.sin` / `tile.cos`；新的组合算子通过 `CompositeLoweringRegistry` 注册降级规则，无需修改 Pass 核心代码。

## 概览 (Overview)

`LowerCompositeOps` 是函数级 (function-level) Pass，对每条 `var = Call(...)` 形式的 `AssignStmt`，若其被调对象在注册表中存在降级规则，则将其改写为一个 `SeqStmts`。对 `tile.sin` / `tile.cos`，规则会发射固定形态的基本算子序列：先做 Cody-Waite 区间归约 (range reduction，π 拆成 4 段)，再做 9 次奇多项式 Horner 求值。原始目标 `Var` 仍是最终 `AssignStmt` 的 LHS，因此下游对该名字/身份的引用都保持不变。

该 Pass **仅支持 FP32**。非 FP32 输入会在算子构造时被共享的 `DeduceTileFP32OnlyType` 类型推导器 (deducer) 拒绝（见 `src/ir/op/tile_ops/unary.cpp:94`），因此本降级 Pass 看到的总是良类型的 FP32 操作数，不需要在 dtype 上失败。

对不含 `tile.sin` / `tile.cos` 的程序，Pass 是**结构性 no-op**：所有其他语句都直接走 `IRMutator::VisitStmt_`。展开生成的也只是基本 tile 算子（`tile.muls`、`tile.adds`、`tile.add`、`tile.sub`、`tile.mul`、`tile.cast`），mutator 不会再改写它们，因此 Pass 也是**幂等的 (idempotent)**。

**所需 (Requires)**：无。

**产生 (Produces)**：无。

**失效 (Invalidates)**：无。

空的 `PassProperties` 契约（`include/pypto/ir/transforms/pass_properties.h` 中的 `kLowerCompositeOpsProperties`）反映了这一事实：本 Pass 的降级完全在已有 tile 算子词汇内进行，既不建立任何 `IRProperty`，也不破坏任何 `IRProperty`。

## 运行时机 (When It Runs)

`LowerCompositeOps` 是 `Default` 流水线 `tile_pto_passes` 的**第一个 Pass**（见 `python/pypto/ir/pass_manager.py`），紧跟 `ConvertTensorToTileOps`（位置 12）和 `OptimizeOrchTensors`（位置 13）之后。此时所有 tensor 级三角调用 (`tensor.sin`、`tensor.cos`) 已经被转换注册表 (conversion registry) 改写成 tile 等价物 (`tile.sin`、`tile.cos`)，tile 流水线即将开始 tile-shape 规范化 (canonicalisation)。在 `FlattenTileNdTo2D` 之前完成三角函数降级，可以让本 Pass 与 2D 展平规则解耦——展开生成的所有基本算子在任意 rank 下都有定义良好的语义。

## 架构 (Architecture)

本 Pass 只是 `CompositeLoweringRegistry` 之上的轻量分发器：

```text
include/pypto/ir/transforms/composite_lowering_registry.h
  LoweringBuilder           — 单次调用的暂存区 (Bind + 基本算子构造器)
  CompositeLoweringFn       — (args, span, builder) -> 结果表达式
  CompositeLoweringRegistry — 单例，按算子名分发

src/ir/transforms/lower_composite_ops_pass.cpp
  LowerCompositeOpsMutator  — 遍历函数，对每个 Call 查表

src/ir/transforms/composite_ops/<op>_lowering.cpp
  规则实现 + RegisterXxxLoweringRules(registry)
```

新增一个组合算子的步骤：

1. 在 `src/ir/transforms/composite_ops/<op>_lowering.cpp` 实现规则。规则接收已 visit 过的参数表达式、`Span` 和一个 `LoweringBuilder`，其 `Bind` 助手会为每个中间临时变量追加一条 `AssignStmt`。
2. 暴露 `void Register<Op>LoweringRules(CompositeLoweringRegistry&)`。
3. 在 `composite_lowering_registry.cpp` 的 `CompositeLoweringRegistry::CompositeLoweringRegistry()` 构造函数里调用它。
4. 把源文件加进 `CMakeLists.txt`。

无需修改分发 Pass。

## 算法 (Algorithm，sin / cos 规则)

`src/ir/transforms/composite_ops/sin_cos_lowering.cpp` 中的 `LowerSinCos` 由 `is_cos` 参数化。mutator 重写的是 `VisitStmt_(const AssignStmtPtr&)`，而不是 `VisitCall`，因为每个三角算子要展开成 ~33 条语句，每条都需要新临时 `Var`。在语句级 (statement level) 工作让规则可以通过 builder 直接把语句追加到外围序列里。

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

所有常量均为 FP32 字面量（取自 `src/ir/transforms/lower_composite_ops_pass.cpp:46-61`，与 `gitcode.com/cann/pypto:framework/src/interface/tileop/vector/unary.h` 上游参考实现一致）：

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

连跑两次 `LowerCompositeOps` 会得到与第一次完全相同的 IR：展开后只剩基本算子 (`tile.muls`、`tile.adds`、`tile.add`、`tile.sub`、`tile.mul`、`tile.cast`)，而 mutator 只改写 `tile.sin` / `tile.cos` 的 `Call`，所以第二次访问 body 时不会有任何变化。`tests/ut/ir/transforms/test_lower_composite_ops.py` 中的 `test_sin_lowering_is_idempotent` 与 `test_cos_lowering_is_idempotent` 验证了这一性质。

## 实现要点 (Implementation Notes)

mutator 重写 `VisitStmt_(const AssignStmtPtr&)` 而不是 `VisitCall`，原因是每个三角算子要往外围序列里塞约 33 条语句。如果在 `VisitCall` 内做拼接，需要让一个表达式返回多个表达式，`IRMutator` 并不支持；改在 `VisitStmt_` 里做，`LowerSinCos` 可以直接构建一个 `vector<StmtPtr>`，并视情况返回单条绑定 `AssignStmt` 或新的 `SeqStmts`。

每个中间结果都绑定到一个用 `auto_name::BuildName` 生成的临时 `Var`，base 名取用户给的目标名。整个函数共享一个 `temp_id_` 计数器，确保函数内多个三角调用之间临时名不会冲突。

`tile.cast` 模式 `RINT`（cos）、`ROUND`（sin）、`FLOOR`（sign）、`None`（int↔float）来自 tile 算子注册表的枚举（`src/ir/op/tile_ops/unary.cpp`）。模式选择对正确性至关重要：sin 中 `k` 用 `ROUND` 保持以零为中心对称，使 Horner 多项式看到的 `t` 分布均匀；cos 中 `k` 用 `RINT` 与 `+0.5` 偏移配合，确保偶数 `k` 对应 `π/2` 的偶数倍。

## 相关 (Related)

- **Issue**：[#1289 — Add FP32-only `tile.sin` / `tile.cos` and a lowering pass](https://github.com/hw-native-sys/pypto/issues/1289)。
- **参考实现 (reference implementation)**：`gitcode.com/cann/pypto:framework/src/interface/tileop/vector/unary.h` —— 本 Pass 的常量与算子序列与该上游 CANN/PyPTO 实现逐字对应。
- **算子推导器 (op deducer)**：`src/ir/op/tile_ops/unary.cpp:94` 的 `DeduceTileFP32OnlyType` —— 在算子构造时强制 FP32-only。
- **转换注册表 (conversion registry)**：`src/ir/transforms/op_conversion_registry.cpp` 中的 `RegisterSimple("tensor.sin", "tile.sin")` 与 cos 对应项 —— 上游 tensor-to-tile 改写，产出本 Pass 消费的 `tile.sin` / `tile.cos` 调用。
- **测试**：`tests/ut/ir/transforms/test_lower_composite_ops.py`（结构）与 `tests/ut/ir/transforms/test_lower_composite_ops_numerical.py`（NumPy 数值对照）。
