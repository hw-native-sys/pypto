# DeriveCallDirections Pass

对每个 `Function` body 运行的单阶段 pass：它基于被调用方的 `ParamDirection` 和缓冲区血缘，为每个跨函数 `Call` 推导每个参数的 `ArgDirection`，并将解析后的向量写入 `Call.attrs["arg_directions"]`。它**不**触碰 manual scope 的依赖边。

## Overview

PyPTO 采用**两层方向模型**（在 commit `c53dac0d` 引入）：

- `ParamDirection`（`In` / `Out` / `InOut`）位于被调用方 `Function` 上，描述函数签名契约——*"我读/写这个参数。"*
- `ArgDirection`（`Input` / `Output` / `InOut` / `OutputExisting` / `NoDep` / `Scalar`）位于每个 `Call` 站点上，描述运行时任务提交语义——*"这次提交建立这些依赖、使用这种内存所有权模型。"*

两层必须一致但不相同：在 `DeriveCallDirections` 下，被调用方的 `Out` 参数在 call 站点上可能变成 `OutputExisting` 或 `InOut`，取决于是否已有其它 writer 触碰过同一缓冲区。`ArgDirection::Output` 保留给运行时应分配全新输出缓冲区的显式填充 call 站点；本 pass 永不推断它。

`DeriveCallDirections` 就是连接两层的 pass。它遍历每个 `Function` body 中的所有非 builtin `Call`，并将解析后的每参数向量写入 `Call.attrs["arg_directions"]`（保留键 `kAttrArgDirections`，值类型为 `std::vector<ArgDirection>`）。下游消费者——orchestration 代码生成和运行时任务提交层——直接读取 `Call.attrs["arg_directions"]`，而不是从原始参数方向重新计算。

**Manual scope 的依赖边不归本 pass 管。** 在 `with pl.manual_scope():` 区域内，用户声明的 `pl.submit(..., deps=[...])` 边由 **parser 直接**写入 `Call.attrs["manual_dep_edges"]`（一个 `vector<VarPtr>`，元素为 `Scalar[TASK_ID]` 变量或 `Array[N, TASK_ID]` carry）。没有任何 pass 合成或降级它们；`DeriveCallDirections` 只读写 `arg_directions`。

**何时使用**：在 tile pipeline 稳定后运行（要求 `SplitIncoreOrch`），并在任何观察 `Call.attrs["arg_directions"]` 的消费者之前。在 `Default` 策略中它位于 `FuseCreateAssembleToSlice` 与最后一次 `Simplify` 之间。

## Properties

| Required | Produced | Invalidated |
| -------- | -------- | ----------- |
| `SplitIncoreOrch` | `CallDirectionsResolved` | — |

`CallDirectionsResolved` property 由已注册的 `CallDirectionsResolved` property verifier 校验（工厂 `CreateCallDirectionsResolvedPropertyVerifier()` 位于 `src/ir/verifier/verify_call_directions.cpp`），因此流水线在本 pass 运行后会自动检查所产生的 `arg_directions` 的完整性——不存在单独的 verify pass。参见 [Verifier](99-verifier.md)。

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::DeriveCallDirections()` | `passes.derive_call_directions()` | Program-level |

**工厂函数**：

```cpp
Pass DeriveCallDirections();
```

**Python 用法**：

```python
from pypto.pypto_core import passes

derive_pass = passes.derive_call_directions()
program_with_dirs = derive_pass(program)
```

## Algorithm

该 pass 是一个 `ProgramPass`。对每个 `Function` body 运行三个子阶段。

### 1. 缓冲区根收集

`BufferRootCollector`（定义于 `include/pypto/codegen/orchestration/orchestration_analysis.h`）遍历函数 body，将每个 `Var*` 映射到拥有其底层缓冲区的 `Var*`，并通过赋值、循环和 call 输出传播根标识。该 pass 还从函数的形参构建 `param_vars` 集合，用于快速判断 *"根是否落在某个函数形参上？"*。

### 2. 先前 writer 分析

`PriorWriterCollector` 按 `(Call, local-root)` 判断该 call 是否是该根在其所属作用域内的*首个 writer*。它分两阶段运行：

1. **自底向上缓存**（`PrecomputeWrittenRoots`）：对每个子树，缓存其中任意非 builtin `Call` 所写的本地分配根的并集。当该子树作为外层作用域的兄弟节点出现时，此结果即为它的 *writer footprint*。
2. **自顶向下扫描**（`AnalyzeScope`）：遍历 IR，维护一个 `seen_roots` 集合记录已被先前兄弟写过的根。对每个 `Call`，每个根*不*在 `seen_roots` 中的被调用方 `Out` 参数都记为首个 writer。每个 `ForStmt`（无论 `ForKind`）/ `WhileStmt` / `IfStmt` 进入时使用 `seen_roots` 的*快照拷贝*（使单元内部的写不泄漏到兄弟跟踪），并视作不透明的 writer 单元；`ScopeStmt` 和 `SeqStmts` 共享同一个 `seen_roots`。

### 3. 方向重写

`CallDirectionMutator` 遍历每个非 builtin `Call`。对 Group/Spmd 被调用方，通过 `ComputeGroupEffectiveDirections`（`orchestration_analysis.h`）恢复每个位置的有效方向；其它被调用方使用其声明的 `param_directions_`。`sequential_depth_` 计数器在非 `Parallel` 的 `For` 和 `While` 上递增，驱动下面的 *R-seq* 提升。

对每个位置参数，mutator 按下表挑选方向。被调用方的 `Out` 依次尝试三条提升规则——R-seq → R-prior → R-enclosing；都不触发时保持 `OutputExisting`：

| 被调用方 `ParamDirection` | 实参 | `sequential_depth > 0`？ | 作用域内有先前 writer？ | 所根植的外层参数为 `InOut`？ | 结果 |
| ------------------------- | ---- | ------------------------ | ----------------------- | ---------------------------- | ---- |
| any | 非 tensor | — | — | — | `Scalar` |
| `In` | tensor | — | — | — | `Input` |
| `InOut` | tensor | — | — | — | `InOut` |
| `Out` | tensor | 是 (R-seq) | — | — | `InOut` |
| `Out` | tensor | 否 | 是 (R-prior) | — | `InOut` |
| `Out` | tensor | 否 | 否 | 是 (R-enclosing) | `InOut` |
| `Out` | tensor | 否 | 否 | 否 | `OutputExisting` |

**R-seq** 在顺序循环内保持跨迭代的 write-after-write 链：只要被调用方的 `Out` 处于任意顺序祖先之下，就**无条件**提升为 `InOut`。早期曾有一个"变 offset store 视为不相交"的例外——当被调用方的 `tile.store` offset 依赖某个参数时，把这类调用保留为 `OutputExisting`——该例外已被移除：要 sound 地证明跨迭代写入互不相交，需要一套真正的依赖分析（仿射 offset 抽取、步长与 tile extent 对比、offset 单射性、跨过程组合），而它当时用的廉价语法检查可能悄悄丢掉真实的 WAW 边。**R-prior** 在同一作用域内某个更早的 writer 单元已触碰过同一根时，保持跨兄弟的 WAW 依赖。**R-enclosing** 当实参根植的外层函数参数被显式声明为 `pl.InOut` 时，遵从该声明。

预填充的 `Call.attrs["arg_directions"]` 被视作权威并保持不动（某些方向如 `NoDep` 无法从结构上推导）。`Call` 构造函数的 `ValidateArgDirectionsAttr` 仅在向量非空时强制 arity；空向量仍可附加，并会在之后被 `CallDirectionsResolved` verifier 拒绝。

**幂等性**：mutator 跳过任何已带 `attrs["arg_directions"]`（`HasArgDirections()`）的 call，因此第二次运行不会改动已解析的 call。两次运行该 pass 因而产生结构上完全相同的 IR（由 `TestDeriveIdempotent::test_idempotent` 回归测试）。

## Example

在顶层连续两次写同一本地分配缓冲区的 call。第一次是唯一的 writer 单元，保持 `OutputExisting`；第二次命中 R-prior，被提升为 `InOut`，使运行时保持对 `local` 的跨 call WAW 依赖。

### Before

```python
@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[64], pl.FP32],
        out: pl.Out[pl.Tensor[[64], pl.FP32]],
    ) -> pl.Tensor[[64], pl.FP32]:
        t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
        ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
        return ret

    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        local: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
        local = self.kernel(x, local)   # arg_directions = []  (pre-pass)
        local = self.kernel(x, local)   # arg_directions = []  (pre-pass)
        return local
```

### After

```python
# Same IR shape; only Call.attrs["arg_directions"] changes:
local = self.kernel(x, local)   # arg_directions = [Input, OutputExisting]
local = self.kernel(x, local)   # arg_directions = [Input, InOut]
```

`kernel` 被调用方为参数 `out` 声明 `Out`。因为 `local` 是本地分配的（根落在 `pl.create_tensor` 而非 `main` 形参上），第一次 call 得到 `OutputExisting`（无顺序祖先、无先前 writer 单元），第二次在同一作用域内看到先前 writer 而被提升为 `InOut`。

## Implementation

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass DeriveCallDirections();
```

**Properties**：`include/pypto/ir/transforms/pass_properties.h`

```cpp
inline const PassProperties kDeriveCallDirectionsProperties{
    .required = {IRProperty::SplitIncoreOrch},
    .produced = {IRProperty::CallDirectionsResolved}};
```

**实现**：`src/ir/transforms/derive_call_directions_pass.cpp`

- `PriorWriterCollector` —— 每作用域的首 writer 分析（自底向上缓存 + 自顶向下扫描）
- `CallDirectionMutator` —— `IRMutator`，用解析后的 `arg_directions` 向量重写每个非 builtin `Call`
- 复用 `include/pypto/codegen/orchestration/orchestration_analysis.h` 中的 `BufferRootCollector` 与 `ComputeGroupEffectiveDirections`

**Property verifier**：`src/ir/verifier/verify_call_directions.cpp`（工厂位于 `include/pypto/ir/verifier/verifier.h`）

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("derive_call_directions", &pass::DeriveCallDirections,
           "Derive Call attrs['arg_directions'] from callee param directions and buffer lineage.");
```

**Type stub**：`python/pypto/pypto_core/passes.pyi`

**手写 IR 辅助**：`python/pypto/ir/directions.py`（`make_call`、小写别名）—— 供测试和手工构建的 IR 片段在 pass 运行前附加显式方向使用。

**Tests**：`tests/ut/ir/transforms/test_derive_call_directions.py`

- `TestDeriveDirectionMatrix` —— 对 (callee_dir, origin) → ArgDirection 映射表的每个单元各一个测试，包括 R-seq（`pl.range`、`while`）和 R-prior（顶层 + 分支 / 顶层之后的 parallel）边界情形
- `TestDeriveIdempotent` —— 两次运行该 pass 得到结构相等的 IR
- `TestDerivePreservesExplicit` —— 预填充的 `arg_directions` 不被覆盖
- `TestVerifyPositive` / `TestVerifyNegative` —— `CallDirectionsResolved` property verifier 接受该 pass 的输出，并拒绝格式错误的 `arg_directions` 赋值
