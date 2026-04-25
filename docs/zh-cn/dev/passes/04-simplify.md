# Simplify Pass

使用代数重写规则和边界分析，折叠算术表达式、类型中嵌入的 shape 表达式以及标量常量绑定。

## 概述

`Simplify` 是一个函数级 Pass，依托 `arith::Analyzer` 就地重写 IR，主要做三类工作：

1. **算术折叠**：在每个表达式叶子上执行（例如 `x + 0 → x`、`x * 1 → x`、`min(a, a) → a`，以及分析器能判定的比较）。
2. **类型重建**：重新遍历 `TensorType`、`TileType`、`TupleType` 中嵌入的 shape 表达式，使内存中的 IR 与重新解析得到的结果一致。
3. **标量常量传播 + DCE**：当一个标量 `Var` 仅被赋值一次且赋的是常量时，将该常量绑定到分析器并向所有下游使用处传播；随后用一个保守的标量 DCE 把已经死掉的绑定本身删除。

在 `pass_manager.py` 的 `Default` 策略中本 Pass 运行**两次**：

- **SSA 后**（在 `ConvertToSSA` 之后、`FlattenCallExpr` 之前）：将闭包捕获的常量（如 `CHUNK_K: Scalar[INDEX] = 512`）传播进 shape 表达式与类型，使后续的 tile lowering Pass 看到的是字面量而不是变量。
- **tile pipeline 末尾**（在 `DeriveCallDirections` 之后）：清理由内存空间推断、layout 解析等晚期 lowering 暴露出来的可折叠表达式。

**需要 (Requires)**：无。

**产生 (Produces)**：无。

**失效 (Invalidates)**：无。

`PassProperties` 为空（`include/pypto/ir/transforms/pass_properties.h` 中的 `kSimplifyProperties`）是有意为之：Simplify 足够保守，会保留调用方此前可能已经建立的所有属性（`SSAForm`、`NormalizedStmtStructure`、`IncoreTileOps` 等）——它只重写表达式、删除标量绑定，从不改变语句结构。

## 使用时机

- 在 SSA 转换之后、tile pipeline 检查类型/shape 之前，把标量常量传播进去。
- 在 tile pipeline 末尾作为清理 Pass，确保下游产物（打印的 IR、codegen）不会残留 `K + 0` 或 `idx * 1` 这类痕迹。
- 任何会产生新表达式的 Pass 之后；Simplify 代价低且幂等，可以放心地防御性地插入。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::Simplify()` | `passes.simplify()` | 函数级 |

**工厂函数**：

```cpp
Pass Simplify();
```

**Python 用法**：

```python
from pypto.pypto_core import passes

simplify_pass = passes.simplify()
program_simplified = simplify_pass(program)
```

## 算法

由 `src/ir/transforms/simplify_pass.cpp` 中的 `TransformSimplify` 分五个阶段实现：

1. **多次赋值收集**：`MultiAssignCollector` 遍历函数体，记录所有「被多次赋值」或「在嵌套 `IfStmt`/`ForStmt`/`WhileStmt` 体内被赋值」的标量 `Var`。这些 `Var` 不会被绑定为常量，避免某次过期的初始值越过后续的重新赋值传播出去。在 SSA 形式下该检查是冗余的，但保留以便对 pre-SSA 调用方仍然安全。
2. **`SimplifyMutator` 遍历**：继承自 `arith::IRMutatorWithAnalyzer`。分析器维护一个约束栈（循环变量边界、if 分支条件、标量绑定）。折叠发生在叶子节点而非仅顶层表达式，因为分析器顶层的 `Simplify` 不会递归进入非算术容器（`Call`、`MakeTuple`）：
   - `VarPtr`：先按变量重映射表替换，再交给分析器化简。
   - `BinaryExpr` / `UnaryExpr`：先访问子节点，再折叠重建后的节点。
   - `CallPtr`：刷新结果 `type_`，让 shape 参数被折叠后的 Call 与重新解析得到的 Call 在结构上相等。
   - `AssignStmt`：当 LHS 类型是 `ScalarType`、RHS 是 `ConstInt`/`ConstFloat`/`ConstBool`，且 LHS 不在 `multi_assigned_` 中时，把 LHS `Var` 绑定到化简后的 RHS。
   - `ForStmt`：在访问循环体前重建 `iter_args_`，使体内的引用对应到新的标识；如果 `start_` 与 `stop_` 都折叠为 `ConstInt` 且 `stop > start`，则在访问循环体期间把循环变量绑定到这一区间，退出时解绑；在访问体之后重建 `return_vars_`，让体内发现的折叠也反映到返回类型中。
   - `IfStmt`：进入 `Analyzer::GetConstraintContext(cond)` 处理 then 分支，进入 `Not(cond)` 处理 else 分支。
   - `SpmdScopeStmt`：折叠 `core_num_`（如 `MAX // TILE` 这样的闭包算术，可能需要 SSA 之后再化简一次）。
3. **类型重建**：`SimplifyType` 递归地处理 `TensorType`、`TileType`、`TupleType`，对每一个嵌入的表达式（shape、stride、valid_shape、start_offset、view 字段）调用 `SimplifyExpr`。当无变化时保留原对象，使往返一致性检查仍然便宜。
4. **标量 DCE**：mutator 完成后，`dce::EliminateDeadScalarAssignments` 在展平的函数体上运行，删除所有「全部使用都被折掉了」的标量 `AssignStmt`。该 DCE 是保守的：永远不会删除 Call 支撑的赋值，因为 IR 目前还没有纯度标注，`Call` 可能存在可观察的副作用。
5. **循环状态修复**：如果 DCE 删除了任何语句，由 `loop_repair::MakeBody` 重新组装函数体，确保循环携带元信息（yield/return 映射）保持一致。

## 示例

### 代数恒等式

**变换前**：

```python
def main(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    a = x + 0
    b = a * 1
    return b
```

**变换后**：

```python
def main(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x
```

`x + 0 → x` 和 `x * 1 → x` 在每个算术叶子上生效。两个标量绑定随后被 DCE 阶段删除，函数体收敛到 return。

### 循环边界感知的折叠

**变换前**：

```python
for i in pl.range(0, 8):
    if i < 16:
        body(i)
```

**变换后**：

```python
for i in pl.range(0, 8):
    body(i)
```

在访问循环体期间，分析器被告知 `i ∈ [0, 8)`。条件 `i < 16` 因此折叠为 `True`，`IfStmt` 收敛到其 then 分支，外层 `for` 保持不变。

### 标量常量传播 + DCE

**变换前**（`ConvertToSSA` 之后，闭包值 `CHUNK_K = 512`）：

```python
CHUNK_K__ssa_v0: pl.Scalar[pl.INDEX] = 512
acc: pl.Tile[[CHUNK_K__ssa_v0, 64], pl.FP32] = tile.zeros(...)
for k in pl.range(0, K, CHUNK_K__ssa_v0):
    body(k)
return acc
```

**变换后**：

```python
acc: pl.Tile[[512, 64], pl.FP32] = tile.zeros(...)
for k in pl.range(0, K, 512):
    body(k)
return acc
```

`CHUNK_K__ssa_v0` 在其 `AssignStmt` 处被绑定到 `512`。所有下游引用——包括 `acc` 的 `TileType` 中嵌入的 shape——都在类型重建阶段折叠为字面量。已经死掉的绑定随后被 DCE 阶段删除。这正是「SSA 后」这一调度点的主要动机：诸如 `FlattenTileNdTo2D`、`InferTileMemorySpace` 等 tile lowering Pass 看到的将是具体的 shape 字面量，而不是不透明的标量 `Var`。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass Simplify();
```

**属性**：`include/pypto/ir/transforms/pass_properties.h`

```cpp
inline const PassProperties kSimplifyProperties{};
```

**实现**：`src/ir/transforms/simplify_pass.cpp`

- `MultiAssignCollector` —— IRVisitor，标记不安全绑定为常量的标量 `Var`。
- `SimplifyMutator` —— 继承自 `arith::IRMutatorWithAnalyzer`；在叶子上折叠表达式，并在 `Var` / `IterArg` 嵌入的 shape 表达式简化时重建其类型。
- `TransformSimplify` —— 编排五个阶段（收集 → 变换 → 类型重建 → DCE → 修复），仅在函数体确实变化时返回新的 `Function`。

**底层分析器**：`src/ir/arith/analyzer.cpp`、`src/ir/arith/ir_mutator_with_analyzer.cpp`。分析器组合了一个重写化简器、常量区间边界分析器、传递性比较分析器和一个约束栈。

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def(
    "simplify", &pass::Simplify,
    "Create a pass that simplifies expressions and statements using algebraic rules and bound analysis");
```

**类型存根**：`python/pypto/pypto_core/passes.pyi`

```python
def simplify() -> Pass:
    """Create a pass that simplifies expressions and statements using algebraic rules and bound analysis."""
```

**测试**：`tests/ut/ir/transforms/test_simplify_pass.py`

- Pass 元数据（名称为 `"Simplify"`，required/produced 属性集为空）。
- 恒等式化简（`x + 0`、`x * 1`、`min(a, a)` 等）。
- 通过 `Call` 参数和嵌入 shape 表达式的常量折叠。
- 通过 `ForStmt` 分析器绑定实现的循环边界感知折叠。
- 通过 `Analyzer::GetConstraintContext` 实现的 if 分支约束传播。
- SSA 形式下的标量常量传播。
- 保守的标量 DCE —— 仅当所有使用都可折叠时才删除。
