# DeriveCallDirections Pass

基于被调用方的 `ParamDirection` 和缓冲区血缘，为每个跨函数 `Call` 推导每个参数的 `ArgDirection`。

## 概述

PyPTO 采用**两层方向模型**（在提交 `c53dac0d` 中引入）：

- `ParamDirection`（`In` / `Out` / `InOut`）位于被调用方 `Function` 上，描述函数签名约定——*"我读取/写入这个参数"*。
- `ArgDirection`（`Input` / `Output` / `InOut` / `OutputExisting` / `NoDep` / `Scalar`）位于每个 `Call` 调用点上，描述运行时任务提交语义——*"本次提交建立这些依赖关系并采用这种内存所有权模型"*。

两层必须保持一致，但并不完全相同：就当前 `DeriveCallDirections` 的推导规则而言，被调用方的 `Out` 参数在调用点上会变为 `OutputExisting` 或 `InOut`，取决于该缓冲区是否已被其他写入者触及。`ArgDirection::Output` 仅用于显式填写方向时表达"运行时分配输出缓冲区"的语义，本 pass 不会自动推导出该方向。

`DeriveCallDirections` 就是连接两层的 pass。它遍历每个 `Function` body 中的所有非 builtin `Call`，并将解析后的每参数向量写入 `Call.attrs["arg_directions"]`（保留键 `kAttrArgDirections`，值类型为 `std::vector<ArgDirection>`）。下游消费者——orchestration 代码生成和运行时任务提交层——直接读取该向量，而不是从原始参数方向重新计算。

**何时使用**：在 tile 流水线稳定后（要求满足 `SplitIncoreOrch`）、并在任何观察 `Call.attrs["arg_directions"]` 的消费者之前运行。在 `Default` 策略中，它位于 `FuseCreateAssembleToSlice` 与最终 `Simplify` 之间（文档编号 28，即排除 utility pass 后的第 28 个 pipeline pass；编号规则见 `.claude/rules/pass-doc-ordering.md`）。

## 属性

| Required | Produced | Invalidated |
| -------- | -------- | ----------- |
| `SplitIncoreOrch` | `CallDirectionsResolved` | — |

`CallDirectionsResolved` 属性由 `src/ir/verifier/verify_call_directions.cpp` 中通过 `CreateCallDirectionsResolvedPropertyVerifier()` 注册/创建的 `CallDirectionsResolved` 属性验证器进行验证，因此 pass 运行后流水线会自动检查所产生的 `arg_directions` 完整性——不存在独立的 verify pass。参见[验证器](99-verifier.md)。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::DeriveCallDirections()` | `passes.derive_call_directions()` | Program 级 |

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

## 算法

该 pass 是一个 `ProgramPass`，对每个 `Function` body 运行三个阶段。

### 1. 缓冲区根收集

`BufferRootCollector`（定义在 `include/pypto/codegen/orchestration/orchestration_analysis.h`）遍历函数 body，将每个 `Var*` 映射到拥有其底层缓冲区的 `Var*`，并在赋值、循环和函数调用输出之间传播根标识。pass 还从函数形式参数构建一个 `param_vars` 集合，用于快速判断*"是否根植于函数参数？"*。

### 2. 先前写入者分析

`PriorWriterCollector` 针对每个 `(Call, local-root)` 组合判断该调用是否是其所在作用域内对该 root 的*第一个写入者*。它分两阶段：

1. **自底向上缓存**（`PrecomputeWrittenRoots`）：为每个子树缓存其内部所有非 builtin `Call` 写入的本地分配 root 的并集。该结果在该子树作为外层作用域的兄弟节点出现时，作为它的*写入者足迹*。
2. **自顶向下扫描**（`AnalyzeScope`）：遍历 IR，并维护一个 `seen_roots` 集合，记录已被前置兄弟节点写入的 root。对于每个 `Call`，若其某个被调用方为 `Out` 的实参对应的 root *不在* `seen_roots` 中，则将其记录为第一个写入者。每个 `ForStmt`（不论 `ForKind`）/ `WhileStmt` / `IfStmt` 在进入时使用 `seen_roots` 的*快照副本*（这样单元内部的写入不会泄漏到兄弟跟踪中），并被视为不透明的写入单元；`ScopeStmt` 与 `SeqStmts` 共享同一个 `seen_roots`。

### 3. 方向重写

`CallDirectionMutator` 遍历每个非 builtin `Call`。对于 Group/Spmd 被调用方，通过 `ComputeGroupEffectiveDirections`（`orchestration_analysis.h`）恢复每位置的有效方向；其他被调用方使用其声明的 `param_directions_`。`sequential_depth_` 计数器在非 `Parallel` 的 `For` 和 `While` 上递增，用于驱动下面的 *R-seq* 提升。

对于每个位置参数，mutator 按下表选择方向：

| Callee `ParamDirection` | 实参来源 | `sequential_depth > 0`？ | 作用域内有先前写入者？ | Result |
| ----------------------- | -------- | ------------------------ | ---------------------- | ------ |
| any | 非 tensor | — | — | `Scalar` |
| `In` | tensor | — | — | `Input` |
| `InOut` | tensor | — | — | `InOut` |
| `Out` | 根植于函数参数 | — | — | `OutputExisting` |
| `Out` | 本地缓冲区 | 是 (R-seq) | — | `InOut` |
| `Out` | 本地缓冲区 | 否 | 是 (R-prior) | `InOut` |
| `Out` | 本地缓冲区 | 否 | 否 | `OutputExisting` |

**R-seq** 在顺序循环内保持跨迭代的 write-after-write 链：同一缓冲区槽每次迭代都会被写入一次，因此运行时必须在该槽上序列化迭代。**R-prior** 当同一作用域中较早的写入单元已经触及同一 root 时，保留跨兄弟的 WAW 依赖关系。

预先填充的 `Call.attrs["arg_directions"]` 被视为权威信息并保持不变（像 `NoDep` 这类方向也无法仅从结构上推导得出）。需要注意的是，`Call` 构造函数中的 `ValidateArgDirectionsAttr` 只会在向量非空时检查其长度是否与参数个数匹配；空向量仍可被构造，但随后不会通过 `CallDirectionsResolved` 的属性验证。

**幂等性**：mutator 一旦发现 `attrs["arg_directions"]` 已存在（`HasArgDirections()`）即直接保留原 `Call`，所以第二次运行时已解析的调用不会被改写。因此连续运行该 pass 两次会产生结构上完全相同的 IR（由 `TestDeriveIdempotent::test_idempotent` 回归测试验证）。

## 示例

两个连续调用写入同一本地分配缓冲区。第一个是该作用域内唯一的写入单元，因此保持 `OutputExisting`；第二个触发 R-prior 并被提升为 `InOut`，从而让运行时在 `local` 上保留跨调用的 WAW 依赖关系。

### 之前

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
        local = self.kernel(x, local)   # arg_directions = []  （pass 之前）
        local = self.kernel(x, local)   # arg_directions = []  （pass 之前）
        return local
```

### 之后

```python
# IR 结构相同；仅 Call.attrs["arg_directions"] 发生变化：
local = self.kernel(x, local)   # arg_directions = [Input, OutputExisting]
local = self.kernel(x, local)   # arg_directions = [Input, InOut]
```

被调用方 `kernel` 为参数 `out` 声明了 `Out`。由于 `local` 是本地分配的（根植于 `pl.create_tensor`，而非 `main` 的某个参数），第一个调用得到 `OutputExisting`（无顺序祖先、无先前写入单元），而第二个调用看到同作用域内已有先前写入者，因此被提升为 `InOut`。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass DeriveCallDirections();
```

**属性**：`include/pypto/ir/transforms/pass_properties.h`

```cpp
inline const PassProperties kDeriveCallDirectionsProperties{
    .required = {IRProperty::SplitIncoreOrch},
    .produced = {IRProperty::CallDirectionsResolved}};
```

**实现**：`src/ir/transforms/derive_call_directions_pass.cpp`

- `PriorWriterCollector` —— 每作用域的第一写入者分析（自底向上缓存 + 自顶向下扫描）
- `CallDirectionMutator` —— 一个 `IRMutator`，用解析后的 `arg_directions` 向量重写每个非 builtin `Call`
- 复用 `include/pypto/codegen/orchestration/orchestration_analysis.h` 中的 `BufferRootCollector` 与 `ComputeGroupEffectiveDirections`

**属性验证器**：`src/ir/verifier/verify_call_directions.cpp`（工厂函数声明在 `include/pypto/ir/verifier/verifier.h`）

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("derive_call_directions", &pass::DeriveCallDirections,
           "Derive Call attrs['arg_directions'] from callee param directions and buffer lineage. ...");
```

**类型存根**：`python/pypto/pypto_core/passes.pyi`

**手写 IR 辅助**：`python/pypto/ir/directions.py`（`make_call`、小写别名）—— 用于在 pass 运行前为 IR 片段附加显式方向的测试和手写 IR 代码。

**测试**：`tests/ut/ir/transforms/test_derive_call_directions.py`

- `TestDeriveDirectionMatrix` —— 为 (callee_dir, origin) → ArgDirection 映射表的每个单元各设一个测试，包含 R-seq（`pl.range`、`while`）与 R-prior（顶层 + 分支 / 顶层后跟 parallel）等边界情况
- `TestDeriveIdempotent` —— 两次运行该 pass 产生结构相等的 IR
- `TestDerivePreservesExplicit` —— 预先填充的 `arg_directions` 不会被覆盖
- `TestVerifyPositive` / `TestVerifyNegative` —— `CallDirectionsResolved` 属性验证器接受 pass 输出，并拒绝格式错误的 `arg_directions` 赋值
