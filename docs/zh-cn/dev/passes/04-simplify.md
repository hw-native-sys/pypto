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
   - `ForStmt`：在访问循环体前重建 `iter_args_`，使体内的引用对应到新的标识；如果 `start_` 与 `stop_` 都折叠为 `ConstInt` 且 `stop > start`，则在访问循环体期间把循环变量绑定到这一区间，退出时解绑；在访问体之后重建 `return_vars_`，让体内发现的折叠也反映到返回类型中。纯单次/零次循环还会被原地折叠 —— 见下文「控制流折叠」。
   - `IfStmt`：进入 `Analyzer::GetConstraintContext(cond)` 处理 then 分支，进入 `Not(cond)` 处理 else 分支。可由分析器证明的条件也会被折叠 —— 见下文「控制流折叠」。
   - `SpmdScopeStmt`：折叠 `core_num_`（如 `MAX // TILE` 这样的闭包算术，可能需要 SSA 之后再化简一次）。
3. **类型重建**：`SimplifyType` 递归地处理 `TensorType`、`TileType`、`TupleType`，对每一个嵌入的表达式（shape、stride、valid_shape、start_offset、view 字段）调用 `SimplifyExpr`。当无变化时保留原对象，使往返一致性检查仍然便宜。
4. **标量 DCE**：mutator 完成后，`dce::EliminateDeadScalarAssignments` 在展平的函数体上运行，删除所有「全部使用都被折掉了」的标量 `AssignStmt`。该 DCE 是保守的：永远不会删除 Call 支撑的赋值，因为 IR 目前还没有纯度标注，`Call` 可能存在可观察的副作用。
5. **循环状态修复**：如果 DCE 删除了任何语句，由 `loop_repair::MakeBody` 重新组装函数体，确保循环携带元信息（yield/return 映射）保持一致。

### 控制流折叠

两个折叠在 `SimplifyMutator` 遍历内部运行，因此与周围的表达式级处理共享分析器的约束栈：

- **Fold A —— 常量条件 `IfStmt` 折叠**。条件被化简后，分别用 `CanProve(cond)` 与 `CanProve(Not(cond))` 询问分析器。任一极性被证明，则丢弃死分支并把保留分支提升到父作用域。当 `return_vars_` 非空时，保留分支末尾的 `YieldStmt` 被剥离，每个 `return_vars[i]` 在 `var_remap_` 中绑定到对应的 yielded 值，使后续兄弟语句（以及函数 `ReturnStmt`）直接读取该值。真/假两种极性的处理是对称的；唯一的边界情况是「永远为假，无 else，且 `return_vars_` 为空」，此时折叠为空体。
- **Fold B —— 纯单次/零次 `ForStmt` 折叠**。仅对*纯*顺序循环触发：`chunk_config_` 为空、`attrs_` 为空、`kind_ == ForKind::Sequential`。对这类循环，用 `CanProveGreaterEqual(step, 1)` 加 `CanProve(stop <= start)`（零次）或 `CanProve(start < stop && stop <= start + step)`（一次）询问分析器以证明循环次数。零次时，为每个 return var 发出 `AssignStmt(return_vars[i], iter_args[i].initValue_)` 并丢弃循环体；一次时，用 `DeepClone` 复制循环体并将 `loop_var → start`、`iter_args[i] → init_values[i]` 直接代入，再次访问克隆体让进一步折叠在同一次 Pass 中发生，最后剥离末尾的 `YieldStmt` 并把 `return_vars[i] → yielded_value[i]` 写入 `var_remap_`（与 Fold A 的提升机制一致）。

在循环体上使用 `DeepClone`（而非就地的 `var_remap_` 覆盖）是因为 `clone_def_vars=true` 在这里至关重要：`MultiAssignCollector` 是在折叠之前的 IR 上跑的，可能已经把原循环体内定义的标量 `Var` 标记为多次赋值（保护 pre-SSA 调用方的安全检查）。在每个定义点产生新的 `Var` 克隆能打破这种过期标记，使后续对（例如）`cur_offset = 0` 的使用得以被分析器绑定并向下游传播。

`return_vars` 通过 `var_remap_` 代换而非直接产出 `AssignStmt(rv, yielded)`，这是有意为之：编排（orchestration）代码生成器的角色感知命名消歧（`role == "out"` 等）会把多个 role 标签的 SSA 版本折叠到同一个 C++ 标识符，于是 `out__rv_v2 = out__co_l0_rv_v3` 这样的别名赋值会下沉为不合法的 `auto out = out;`。在使用点代换可以完全绕开消歧。

两种折叠在同一次 Pass 中可以叠加：当 Fold B 把 `loop_var → 0` 代入循环体后，类似 `if loop_var == 0` 的谓词会变成 `if 0 == 0` → `ConstBool(true)`，紧接着就被 Fold A 折掉，无需再跑一次 Simplify。

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

### 常量条件分支（Fold A）

**变换前**：

```python
for i in pl.range(0, 8, 2):
    if i == -1:
        body_dead(i)
    else:
        body_live(i)
```

**变换后**：

```python
for i in pl.range(0, 8, 2):
    body_live(i)
```

分析器在访问循环体期间得知 `i ∈ [0, 8)`。`CanProve(Not(i == -1))` 成功 —— 该比较静态恒为假 —— 因此 Fold A 丢弃 then 分支并把 else 分支提升到外层 for 体。永远为真的条件走对称路径（丢弃 else，提升 then）。当 IfStmt 拥有 `return_vars_` 时，保留分支末尾的 `YieldStmt` 会被改写为对 return vars 的 `AssignStmt`。

### 单次循环折叠（Fold B）

**变换前**：

```python
for ko in pl.range(0, 128, 128):
    if ko == 0:
        first_iter(ko)
    else:
        later_iter(ko)
```

**变换后**：

```python
first_iter(0)
```

`pl.range(0, 128, 128)` 满足循环次数证明 `start < stop && stop <= start + step`，因此 Fold B 通过 `DeepClone` 把 `ko → 0` 代入循环体并提升到父作用域。代换之后内层的 `if ko == 0` 变为 `if 0 == 0`，被 `analyzer_->Simplify` 化简为 `ConstBool(true)`，进而触发 Fold A 丢掉死的 else 分支 —— 两种折叠在同一次 Simplify 中叠加生效。零次循环走相同的路径：为每个 `return_vars[i] = iter_args[i].initValue_` 发出 `AssignStmt`，并整体丢弃循环体。

带有 `chunk_config_`、`attrs_` 或非 `Sequential` `kind_` 的循环会被跳过 —— 这些形式参与执行模型契约（chunk 循环配对、Parallel/Unroll/Pipeline 调度），下游 Pass 可能依赖它们仍然以 `ForStmt` 形式出现。

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
