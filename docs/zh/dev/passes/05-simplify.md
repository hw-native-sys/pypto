# Simplify Pass

使用代数重写规则和边界分析，折叠算术表达式、类型中嵌入的 shape 表达式以及标量常量绑定。

## 概述

`Simplify` 是一个函数级 Pass，依托 `arith::Analyzer` 就地重写 IR，主要做三类工作：

1. **算术折叠**：在每个表达式叶子上执行（例如 `x + 0 → x`、`x * 1 → x`、`min(a, a) → a`，以及分析器能判定的比较）。
2. **类型重建**：重新遍历 `TensorType`、`TileType`、`TupleType` 中嵌入的 shape 表达式，使内存中的 IR 与重新解析得到的结果一致。
3. **标量绑定以辅助折叠 + DCE**：仅被赋值一次的标量 `Var` 会注册到分析器。在函数体顶层赋的常量会被完整绑定，其字面量向所有下游使用处传播；符号值，或循环/分支内部的常量，只贡献一个 `ConstIntBound`——足以折叠 `if expr == 0` 这类恒死的分支守卫，而不会把标量内联到使用点。残留的死绑定随后由保守的标量 DCE 删除。

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

1. **多次赋值收集**：`MultiAssignCollector` 遍历函数体，记录所有被多次赋值的标量 `Var`。这些 `Var` 不会被绑定到分析器，避免某个过期的值越过后续的重新赋值被使用。仅被赋值一次的 `Var`——即使位于循环体或分支内部——也可以安全绑定：`SimplifyMutator` 会把每个绑定限定在赋值所在的区域内（见阶段 2），在离开该区域时解绑。在 SSA 形式下每个 `Var` 都只被赋值一次，因此不会收集到任何 `Var`。
2. **`SimplifyMutator` 遍历**：继承自 `arith::IRMutatorWithAnalyzer`。分析器维护一个约束栈（循环变量边界、if 分支条件、标量绑定）。折叠发生在叶子节点而非仅顶层表达式，因为分析器顶层的 `Simplify` 不会递归进入非算术容器（`Call`、`MakeTuple`）：
   - `VarPtr`：先按变量重映射表替换，再交给分析器化简。
   - `BinaryExpr` / `UnaryExpr`：先访问子节点，再折叠重建后的节点。
   - `CallPtr`：刷新结果 `type_`，让 shape 参数被折叠后的 Call 与重新解析得到的 Call 在结构上相等。
   - `AssignStmt`：对不在 `multi_assigned_` 中的标量 LHS `Var`，把化简后的 RHS 注册到分析器。函数体顶层的 `ConstInt`/`ConstFloat`/`ConstBool` RHS 会被完整绑定（字面量代入下游使用点）；符号 RHS，或循环/分支内部的常量，只贡献一个 `ConstIntBound`，使恒死的分支守卫得以折叠而不会内联该标量。每个绑定都会被记录，以便所在区域的访问器在退出时解绑。
   - `ForStmt`：在访问循环体前重建 `iter_args_`，使体内的引用对应到新的标识；如果 `start_` 与 `stop_` 都折叠为 `ConstInt` 且 `stop > start`，则在访问循环体期间把循环变量绑定到这一区间，退出时解绑；体内绑定的标量在访问结束后解绑；在访问体之后重建 `return_vars_`，让体内发现的折叠也反映到返回类型中。纯单次/零次循环还会被原地折叠 —— 见下文「控制流折叠」。
   - `IfStmt`：进入 `Analyzer::GetConstraintContext(cond)` 处理 then 分支，进入 `Not(cond)` 处理 else 分支；每个分支内绑定的标量会在该分支结束后解绑，以免泄漏到另一分支或越过 `IfStmt`。可由分析器证明的条件也会被折叠 —— 见下文「控制流折叠」。
   - `WhileStmt`：除没有循环边界外与 `ForStmt` 相同 —— 在访问条件与循环体前重建 `iter_args_`，在访问循环体前后快照并恢复 `var_remap_`，随后重建 `return_vars_`，并采用同样的区域化标量解绑方式。先重建 `iter_args_` 是必需的，而非可有可无：`IterArg` 的*使用*与其声明是同一个节点并携带 `initValue_`，因此当分析器改写了 init 之后，基类 `IRMutator` 会在第一处使用点新建一个 `IterArg`。以循环头为准写入 `var_remap_`，可使所有引用都解析到同一个节点；若省略这一步，循环头仍指向旧的 `IterArg`，而体内所有使用都指向一个未定义的克隆节点（表现为 `UseAfterDef` 失败）。
   - `SpmdScopeStmt`：以同样的区域化标量解绑方式访问其语句体，并额外折叠 `core_num_`（如 `MAX // TILE` 这样的闭包算术，可能需要 SSA 之后再化简一次）。
3. **类型重建**：`SimplifyType` 递归地处理 `TensorType`、`TileType`、`TupleType`，对每一个嵌入的表达式（shape、stride、valid_shape、start_offset、view 字段）调用 `SimplifyExpr`。当无变化时保留原对象，使往返一致性检查仍然便宜。
4. **标量 DCE + 死 yield 槽位裁剪**：mutator 完成后，`dce::EliminateDeadScalarAssignments` 在展平的函数体上运行，删除所有「全部使用都被折掉了」的标量 `AssignStmt`。该 DCE 是保守的：永远不会删除 Call 支撑的赋值，因为 IR 目前还没有纯度标注，`Call` 可能存在可观察的副作用。两次标量 DCE 之间由 `dce::EliminateDeadYieldSlots` 裁掉没有任何读者的 yield 槽位——`IfStmt` 中无人使用的 phi `return_vars_[i]`，以及 `ForStmt` / `WhileStmt` 中 `iter_args_[i]`（循环体内读）与 `return_vars_[i]`（循环后读）都未被使用的循环携带槽位——同时删除对应的 `YieldStmt` 槽位。跨两个 scope 复用同一个 Python 局部变量恰好会产生这种死携带：SSA 用前一个 scope 的值给第二个循环做初值，循环体每一轮都覆盖它，两端都没有人读。若保留下来，它会让前一个 scope 的值变成 live-out；对设备 scope 而言，这会迫使外提出的 kernel 返回一个 `Scalar`——参见 [08-outline_incore_scopes.md](09-outline_incore_scopes.md)。
5. **循环状态修复**：如果 DCE 删除了任何语句，由 `loop_repair::MakeBody` 重新组装函数体，确保循环携带元信息（yield/return 映射）保持一致。

### 控制流折叠

两个折叠在 `SimplifyMutator` 遍历内部运行，因此与周围的表达式级处理共享分析器的约束栈：

- **Fold A —— 常量条件 `IfStmt` 折叠**。条件被化简后，分别用 `CanProve(cond)` 与 `CanProve(Not(cond))` 询问分析器。任一极性被证明，则丢弃死分支并把保留分支提升到父作用域。当 `return_vars_` 非空时，保留分支末尾的 `YieldStmt` 被剥离，每个 `return_vars[i]` 在 `var_remap_` 中绑定到对应的 yielded 值，使后续兄弟语句（以及函数 `ReturnStmt`）直接读取该值。真/假两种极性的处理是对称的；唯一的边界情况是「永远为假，无 else，且 `return_vars_` 为空」，此时折叠为空体。
- **Fold B —— 纯单次/零次 `ForStmt` 折叠**。仅对*纯*顺序循环触发：`attrs_` 为空、`kind_ == ForKind::Sequential`。对这类循环，用 `CanProveGreaterEqual(step, 1)` 加 `CanProve(stop <= start)`（零次）或 `CanProve(start < stop && stop <= start + step)`（一次）询问分析器以证明循环次数。零次时，为每个 return var 发出 `AssignStmt(return_vars[i], iter_args[i].initValue_)` 并丢弃循环体；一次时，用 `DeepClone` 复制循环体并将 `loop_var → start`、`iter_args[i] → init_values[i]` 直接代入，再次访问克隆体让进一步折叠在同一次 Pass 中发生，最后剥离末尾的 `YieldStmt` 并把 `return_vars[i] → yielded_value[i]` 写入 `var_remap_`（与 Fold A 的提升机制一致）。

在循环体上使用 `DeepClone` 且 `clone_def_vars=true`（而非就地的 `var_remap_` 覆盖），是为了让展开后的循环体在每个定义点获得全新的 `Var` 标识，与 `LoopUnrollMutator` 保持一致。这样提升后的副本在结构上与原（已丢弃的）循环体相互独立，并使重新访问时能在与外围作用域不同的标识上绑定循环体内的标量。

`return_vars` 通过 `var_remap_` 代换而非直接产出 `AssignStmt(rv, yielded)`，这是有意为之：编排（orchestration）代码生成器的角色感知命名消歧（`role == "out"` 等）会把多个 role 标签的 SSA 版本折叠到同一个 C++ 标识符，于是 `out__rv_v2 = out__co_l0_rv_v3` 这样的别名赋值会下沉为不合法的 `auto out = out;`。在使用点代换可以完全绕开消歧。

#### 逃逸的 return var

代换只能作用于「`var_remap_` 条目仍然有效时被访问到」的使用点，而 `ForStmt`、`WhileStmt`、`IfStmt` 在离开各自的体时都会把 `var_remap_` 恢复到进入前的基线，以免体内的 remap 改写兄弟语句或循环之后的代码。活过这次恢复的使用点会继续指向原始 `Var` —— 而折叠恰好删除了它唯一的定义，形成 `UseAfterDefCheck` 会报告的悬空引用。

`ReturnVarEscapeIndex`（位于 `simplify_pass.cpp` 的前置分析）按折叠点逐一判定。它对函数体做一次遍历，用前序编号标记所有会恢复 `var_remap_` 的作用域，使每个作用域拥有其子树的连续 id 区间 `[id, end)`；于是「`v` 的所有使用点都在作用域 `S` 内」只需两次整数比较。单调递增的 tick 则把使用点与折叠点排序，因此同一作用域内*位于折叠点之前*的读取同样计为逃逸。一次遍历加上每个折叠点 O(1) 的查询，使 Simplify 仍在 O(N log N) 预算之内。

索引中不存在的语句一律回答「不逃逸」，即保持代换。这涵盖了嵌套在 Fold B `DeepClone` 循环体内部的折叠 —— 克隆体的 `Var` 标识在建索引之后才产生。克隆体内的 `Var` 在其外部不可达，因此唯一未覆盖的情形是：克隆体*内部*存在一个恢复作用域，横亘在这样的折叠点与其 return var 的后续使用点之间 —— 只可能出现在 pre-SSA，且不比本索引引入之前的行为更差。为每个克隆体重新建索引可以补上这一点，但嵌套单次循环将因此付出 O(N²) 的遍历代价。

对逃逸的 `return_vars[i]`，`LiftBodyToReturnVars` 不再记录 remap，而是在折叠点产出 `AssignStmt(return_vars[i], yielded_value[i])`。该赋值必须留在被提升的区域*内部*：yielded 值可能引用循环体内的局部 `Var`，无法外提到循环之后；而在 leak 语义下「最后一次迭代最后写入」恰好就是循环后读取所期望的值。

SSA 形式下不存在逃逸：区域内定义的值不会在区域外被引用，且每个使用点都被其定义支配。由于流水线只在 `ConvertToSSA` 之后运行 Simplify（第 5 和第 46 位），该物化路径在流水线中不会触发 —— 它服务于直接对 pre-SSA IR 运行 Simplify 的调用方，而在那里上述别名赋值的顾虑并不成立，因为 SSA 转换仍会在其后运行。

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

### 通过标量边界折叠死分支守卫

**变换前**：

```python
for ob in pl.range(0, 68, 2):
    off: pl.Scalar[pl.INDEX] = ob * 256 + 256
    if off == 0:
        first_chunk(off)
    else:
        later_chunk(off)
```

**变换后**：

```python
for ob in pl.range(0, 68, 2):
    off: pl.Scalar[pl.INDEX] = ob * 256 + 256
    later_chunk(off)
```

分析器在访问循环体期间得知 `ob ∈ [0, 68)`，因此 `off` 的 `AssignStmt` 为 `off` 注册了 `[256, 17408]` 的 `ConstIntBound`。`CanProve(Not(off == 0))` 随后成功，Fold A 丢弃死的 then 分支。`off` 只用于分析、不会被代换，因此保留下来的 `later_chunk(off)` 仍引用该标量。（若折叠后 `off` 不再被使用，标量 DCE 会删除其绑定。）

### 索引边界从何而来

`INDEX` 是所有索引计算的 dtype，而它是**有符号的**——codegen 为其发射 `arith.cmpi slt` 与
`arith.maxsi`。因此仅凭 dtype 无法证明变量的符号，分析器把未绑定的 `INDEX` Var 视为
`[-inf, +inf]`。非负性必须被建立，而不能被假定：

| 来源 | 边界 | 由谁建立 |
| ---- | ---- | -------- |
| 被赋值的标量 | 其 RHS 的区间 | `BindScalarBound`，来自被产生的值 |
| 循环变量 | `[start, stop)`；`stop` 为符号时取 `[start, +inf)` | `IRMutatorWithAnalyzer` 处理 `ForStmt`，要求步长为正 |
| 分支条件 | 该约束的区间 | 该分支作用域内的 `EnterConstraint` |
| block / subblock 内建 op | `[0, +inf)`；block *数量* 为 `[1, +inf)` | 该 op 自身语义，在 `ConstIntBoundAnalyzer` 中 |
| 整个 shape / valid-shape 维度 | `[0, +inf)` | `DimensionSymbolScope`，包裹 write-union 证明 |
| 运行时标量参数 | `[-inf, +inf]` | 无——取值由调用方决定 |

最后三行正是关键区别，且没有一条是关于 `INDEX` 类型的事实。`tile.get_subblock_idx()` 非负，是因为该
op **返回什么**；维度非负，是因为它是元素个数。

**第二条规则止于维度本身。** `DimensionSymbolScope` 只绑定**本身就是裸符号**的维度，绝不深入到复合
表达式内部。字段是 `valid_shape` 只能说明该字段是 extent，并不能说明计算它的每个变量都是 extent：

```python
valid = pl.max(-x, 0)    # 基于有符号运行时标量的合法动态 extent
```

假定 `x >= 0` 会把它折叠为常量 `0`，读起来就是空区域，从而静默缩小结果。offset 则完全不绑定——offset
中的 `max(x, 0)` 是刻意的钳位，其意义恰恰在于 `x` 可能为负，折叠成 `x` 会移动 store 写入的区域。

```python
pos: pl.Scalar[pl.INDEX] = base - 1   # [-1, +inf)
if pos >= 0:                          # 有效守卫，予以保留
    if pos < 8:
        read_row(pos)
```

在一刀切的 `[0, +inf)` 默认区间下，`pos >= 0` 被证明为恒真，Fold A 丢弃外层守卫——只剩下界检查
独自成立，而负的 `pos` 恰好能通过它，带着随后被钳位到第 0 行的索引进入 `read_row`
（issue #2500）。对裸的 `Scalar[pl.INDEX]` 参数同样成立：调用方可以传入 `-1`。

这也是为什么 `ConstIntBound` 的约束作用域在退出时会**恢复**显式的 `[-inf, +inf]` 绑定而不是删除
它——在 extent 规则下，被删除的条目会退回 `[0, +inf)`，而那并非该变量原本的边界。

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

带有 `attrs_` 或非 `Sequential` `kind_` 的循环会被跳过 —— 这些形式参与执行模型契约（Parallel/Unroll/Pipeline 调度），下游 Pass 可能依赖它们仍然以 `ForStmt` 形式出现。

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

- `MultiAssignCollector` —— IRVisitor，标记被多次赋值（不安全绑定）的标量 `Var`。
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
- 通过循环仿射标量的 `ConstIntBound` 折叠死分支守卫。
- 保守的标量 DCE —— 仅当所有使用都可折叠时才删除。
