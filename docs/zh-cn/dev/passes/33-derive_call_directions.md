# DeriveCallDirections Pass

对每个 `Function` body 分两个阶段运行：**Phase 1** 基于被调用方的 `ParamDirection` 和缓冲区血缘，为每个跨函数 `Call` 推导每个参数的 `ArgDirection`；**Phase 2** 将 `with pl.manual_scope():` 区域内用户声明的 `deps=[...]` 边降级为 TaskId 依赖图，使 orchestration 代码生成可以发出 `params.add_dep(<task_id>)`。Phase 2 原本是名为 `DeriveManualScopeDeps` 的独立 pass，现已并入本 pass 作为实现细节；Python API（`passes.derive_call_directions()`）与 Default 策略中的入口位置保持不变。

## 概述

PyPTO 采用**两层方向模型**（在提交 `c53dac0d` 中引入）：

- `ParamDirection`（`In` / `Out` / `InOut`）位于被调用方 `Function` 上，描述函数签名约定——*"我读取/写入这个参数"*。
- `ArgDirection`（`Input` / `Output` / `InOut` / `OutputExisting` / `NoDep` / `Scalar`）位于每个 `Call` 调用点上，描述运行时任务提交语义——*"本次提交建立这些依赖关系并采用这种内存所有权模型"*。

两层必须保持一致，但并不完全相同：就当前 `DeriveCallDirections` 的推导规则而言，被调用方的 `Out` 参数在调用点上会变为 `OutputExisting` 或 `InOut`，取决于该缓冲区是否已被其他写入者触及。`ArgDirection::Output` 仅用于显式填写方向时表达"运行时分配输出缓冲区"的语义，本 pass 不会自动推导出该方向。

`DeriveCallDirections` 就是连接两层的 pass。Phase 1 遍历每个 `Function` body 中的所有非 builtin `Call`，并将解析后的每参数向量写入 `Call.attrs["arg_directions"]`（保留键 `kAttrArgDirections`，值类型为 `std::vector<ArgDirection>`）。Phase 2 仅在函数 body 含有 `RuntimeScopeStmt(manual=true)` 时运行，重写该 manual 作用域内的 kernel 调用，使其暴露 TaskId 类型的依赖边。下游消费者——orchestration 代码生成和运行时任务提交层——直接读取 `Call.attrs["arg_directions"]` 与 `Call.attrs["manual_dep_edges"]`，而不是从原始参数方向重新计算。

**何时使用**：在 tile 流水线稳定后（要求满足 `SplitIncoreOrch`）、并在任何观察 `Call.attrs["arg_directions"]` / `Call.attrs["manual_dep_edges"]` 的消费者之前运行。在 `Default` 策略中，它位于 `FuseCreateAssembleToSlice` 与最终 `Simplify` 之间。

## 属性

| Required | Produced | Invalidated |
| -------- | -------- | ----------- |
| `SplitIncoreOrch` | `CallDirectionsResolved` | — |

`CallDirectionsResolved` 属性由 `src/ir/verifier/verify_call_directions.cpp` 中通过 `CreateCallDirectionsResolvedPropertyVerifier()` 注册/创建的 `CallDirectionsResolved` 属性验证器进行验证，因此 pass 运行后流水线会自动检查所产生的 `arg_directions` 完整性——不存在独立的 verify pass。参见[验证器](99-verifier.md)。

Phase 2 不引入新的 IRProperty：codegen 直接读取 `Call.attrs["manual_dep_edges"]`。如果程序里没有任何 `RuntimeScopeStmt(manual=true)`，Phase 2 不做任何事。

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

该 pass 是一个 `ProgramPass`。对每个 `Function` body：始终运行 **Phase 1（arg directions）**；仅当 body 含有 `RuntimeScopeStmt(manual=true)` 时才会接着运行 **Phase 2（manual scope 降级）**。

## Phase 1 —— Arg directions

Phase 1 对每个 `Function` body 运行三个子阶段。

### 1.1 缓冲区根收集

`BufferRootCollector`（定义在 `include/pypto/codegen/orchestration/orchestration_analysis.h`）遍历函数 body，将每个 `Var*` 映射到拥有其底层缓冲区的 `Var*`，并在赋值、循环和函数调用输出之间传播根标识。pass 还从函数形式参数构建一个 `param_vars` 集合，用于快速判断*"是否根植于函数参数？"*。

### 1.2 先前写入者分析

`PriorWriterCollector` 针对每个 `(Call, local-root)` 组合判断该调用是否是其所在作用域内对该 root 的*第一个写入者*。它分两阶段：

1. **自底向上缓存**（`PrecomputeWrittenRoots`）：为每个子树缓存其内部所有非 builtin `Call` 写入的本地分配 root 的并集。该结果在该子树作为外层作用域的兄弟节点出现时，作为它的*写入者足迹*。
2. **自顶向下扫描**（`AnalyzeScope`）：遍历 IR，并维护一个 `seen_roots` 集合，记录已被前置兄弟节点写入的 root。对于每个 `Call`，若其某个被调用方为 `Out` 的实参对应的 root *不在* `seen_roots` 中，则将其记录为第一个写入者。每个 `ForStmt`（不论 `ForKind`）/ `WhileStmt` / `IfStmt` 在进入时使用 `seen_roots` 的*快照副本*（这样单元内部的写入不会泄漏到兄弟跟踪中），并被视为不透明的写入单元；`ScopeStmt` 与 `SeqStmts` 共享同一个 `seen_roots`。

### 1.3 方向重写

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

## Phase 2 —— Manual scope 降级

Phase 2 实现为公开工具函数 `LowerManualDepsToTaskId(StmtPtr body) -> StmtPtr`（头文件 `include/pypto/ir/transforms/utils/lower_manual_deps_to_task_id.h`，实现 `src/ir/transforms/utils/lower_manual_deps_to_task_id.cpp`），由 `DeriveCallDirections` 在 Phase 1 完成后调用。此前 Phase 2 是名为 `DeriveManualScopeDeps` 的独立流水线 pass；将降级逻辑抽取到该工具后，流水线可以在同一函数 body 上连续运行 Phase 1 与 Phase 2，无需再做一次全程序遍历。

PyPTO 中 orchestrator 的依赖跟踪有两种作用域：

- **Auto scope**（默认 `PTO2_SCOPE()`）：runtime 根据缓冲区读写重叠（OverlapMap）自动跟踪依赖。
- **Manual scope**（`with pl.manual_scope():` → `RuntimeScopeStmt(manual=true)`）：用户全权负责排序。runtime 跳过 OverlapMap，**所有依赖边都必须由用户通过 `kernel(..., deps=[var, ...])` 显式声明**。本 pass 不会从数据流自行推导任何边——之前的自动数据流推导路径已被删除，因为只要某缓冲区在多个无关 kernel 间被原地复用，它就会产生误报。

`LowerManualDepsToTaskId` 对每个函数 body 顺序运行四个子阶段。

### 2.1 `ManualDepResolveMutator`

对 manual 作用域内的每个 kernel `Call`，把 `Call.attrs["user_manual_dep_edges"]`（DSL 写 `deps=[var, ...]` 时由 parser 写入的 Tensor-Var 边）复制到 `Call.attrs["manual_dep_edges"]`。复制时去重并保持用户原始顺序。从 data-flow 自动推导 dep 边的能力已被有意移除（它会过度串行化共享 `Out` 参数的并行 kernel）；现在用户通过 `deps=[...]` 显式控制依赖集合。每次 submit 的 16 个显式 dep 上限（`PTO2_MAX_EXPLICIT_DEPS`）在 orchestration codegen 阶段统一校验，超过时抛出 `pypto::ValueError`。

### 2.2 `TaskRelevantVarCollector`（闭包分析）

从每个 `kAttrManualDepEdges` 集合中提到的 Tensor Var 出发，把"需要 TaskId 同伴"的标记沿以下边传播：

- **Var 别名**（`b = a` AssignStmt 与 `b = tuple[i]` TupleGetItem 解包）。
- **`ForStmt.iter_args` ↔ `init_value`**（如果 iter_arg 的初值来自被标记的 Var，则其本身也需要 TaskId carry，反之亦然）。
- **`ForStmt.return_vars` ↔ `iter_args`**（带 TaskId 的 iter_arg 产生的 return var 也带 TaskId）。
- **`YieldStmt` 源 ↔ 目标**（双向均需要：`deps=[<iter_arg>]` 从 dest 流向 src；`deps=[<kernel_lhs>]` 从 src 流向 carry destination）。

不动点闭包构建三个集合：`needs_tid_`（需要同伴的所有 Var）、`kernel_lhs_`（作为 user kernel Call 的 LHS 的 Var，使用 `system.task_id_of` 合成路径）、`import_vars_`（在 `needs_tid_` 中且没有 AssignStmt 定义的 Var，典型情况是作为 iter_arg 初值的函数参数）。

### 2.3 `PreallocateTaskIdVars`

为 `needs_tid_` 中的每个 Var 分配一个 TaskId 同伴：

- 普通 `Var`（非 IterArg，例如 kernel LHS 或函数参数）→ 一个名为 `<name_hint>__tid`、类型为 `ScalarType(DataType::TASK_ID)` 的新 `Var`。
- `IterArg` → 一个带相同后缀名的新 `IterArg`；其初值挂到外层 Var 同伴上（通过部分构建的 `tid_map_` 查询）。嵌套循环里，这要求外层同伴先存在，所以 IterArg 分配子阶段会在循环内不动点扫描：初值同伴尚未分配的 iter_arg 会被重新尝试直到链条收敛。

`tid_map_: const Var* → VarPtr` 是同伴身份的唯一来源；其他子阶段都通过它查找以避免指针身份漂移。

### 2.4 `TaskIdLoweringMutator`（IR mutation）

一次 IRMutator 扫描在函数 body 上安装 TaskId 基础设施：

- 对每个 LHS 在 `needs_tid_` 中的 kernel `Call` AssignStmt，紧随其后插入 `<lhs>__tid = system.task_id_of(<lhs>)`。
- 对每个 LHS 在 `needs_tid_` 中的 `tensor.create` AssignStmt（无前驱 task 的占位缓冲区），紧随其后插入 `<lhs>__tid = system.task_invalid()`。
- 对每个普通的 Var 别名 AssignStmt（`b = a`），插入 `b__tid = a__tid`。
- 对每个 TupleGetItem AssignStmt（`b = tuple[i]`），插入 `b__tid = tuple_var__tid`（所有解包元素共享 tuple-producing call 的 task id）。
- 对每个 kernel `Call`，将 `kAttrManualDepEdges` 中的 Tensor Var 改写为对应 TaskId 同伴，并通过 `kAttrTaskIdVar` 指向 LHS 的同伴（这样后续兄弟节点的 `deps=[lhs]` 可以直接通过该属性查到，而无需重跑闭包）。
- 对 manual 作用域内的每个 `ForStmt`，为 `needs_tid_` 中的每个 iter_arg 追加一个 TaskId iter_arg 与 return-var 同伴。yield 值列表对称延长。
- 对 `import_vars_`（用作 TaskId iter_arg 种子的函数参数），在函数 body 入口处先插入 `<param>__tid = system.task_invalid()` AssignStmt，使同伴具有 codegen 可引用的 SSA 定义。

kernel-Call 的改写把**降级后的形式**写入 `kAttrManualDepEdges`（TaskId Var）。codegen 消费该属性；`kAttrUserManualDepEdges` 中的原始 Tensor-Var 形式则保留供 round-trip 打印使用。

## 示例

### Phase 1 —— Arg directions

两个连续调用写入同一本地分配缓冲区。第一个是该作用域内唯一的写入单元，因此保持 `OutputExisting`；第二个触发 R-prior 并被提升为 `InOut`，从而让运行时在 `local` 上保留跨调用的 WAW 依赖关系。

#### 之前

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

#### 之后

```python
# IR 结构相同；仅 Call.attrs["arg_directions"] 发生变化：
local = self.kernel(x, local)   # arg_directions = [Input, OutputExisting]
local = self.kernel(x, local)   # arg_directions = [Input, InOut]
```

被调用方 `kernel` 为参数 `out` 声明了 `Out`。由于 `local` 是本地分配的（根植于 `pl.create_tensor`，而非 `main` 的某个参数），第一个调用得到 `OutputExisting`（无顺序祖先、无先前写入单元），而第二个调用看到同作用域内已有先前写入者，因此被提升为 `InOut`。

### Phase 2 —— 单条 manual dep 边

```python
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64], pl.FP32],
         out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
    with pl.manual_scope():
        scratch = self.stage1(x, scratch)
        out = self.stage2(scratch, out, deps=[scratch])
    return out
```

pass 之后：

```python
scratch__ssa_v0__tid: pl.Scalar[pl.TASK_ID] = self.system.task_invalid()  # import var 种子
with pl.manual_scope():
    scratch__ssa_v5 = self.stage1(x, scratch)
    scratch__ssa_v5__tid: pl.Scalar[pl.TASK_ID] = self.system.task_id_of(scratch__ssa_v5)
    out__ssa_v7 = self.stage2(scratch__ssa_v5, out, deps=[scratch__ssa_v5__tid])
```

codegen 从被重写的 dep 边发出 `params_t1.add_dep(scratch__ssa_v5__tid);`。

### Phase 2 —— 多条 deps + loop carry

```python
with pl.manual_scope():
    for phase in pl.range(N_PHASES):
        for branch in pl.parallel(N_BRANCHES):
            row = (phase * N_BRANCHES + branch) * TILE_M
            out = self.kernel_stripe(data, row, 1.0, out, deps=[out])
```

`out` 在每次迭代被重新绑定，所以两个 ForStmt 都会附加一个 TaskId iter_arg（携带上一次迭代的 task id）。pass 之后：

```python
for phase, (out__iter_v1, out__iter_v1__tid) in pl.range(4, init_values=(out, out__ssa_v0__tid)):
    for branch, (out__iter_v3, out__iter_v3__tid) in pl.parallel(4, init_values=(out__iter_v1, out__iter_v1__tid)):
        out__ssa_v5 = self.kernel_stripe(..., deps=[out__iter_v3__tid])
        out__ssa_v5__tid = self.system.task_id_of(out__ssa_v5)
        out__rv_v4, out__rv_v4__tid = pl.yield_(out__ssa_v5, out__ssa_v5__tid)
    out__rv_v2, out__rv_v2__tid = pl.yield_(out__rv_v4, out__rv_v4__tid)
```

当 trip count 静态已知时，orchestration codegen 把 `pl.parallel` 的 TaskId iter_arg 视为 **大小 `N_BRANCHES` 的数组 carry**：分配 `PTO2TaskId arr[N_BRANCHES]`，每次迭代 yield 写入一个槽，下游消费者按槽位获得一次 `add_dep`。这保证 phase-fence 对**所有**并行迭代都生效，而不仅是最后一次派发的迭代。容量上限同 `PTO2_MAX_EXPLICIT_DEPS = 16`；常量 trip count 超过该值时 codegen 报错；`pl.parallel` 下携带 manual dep 的非常量 trip count 也会在 codegen 时拒绝并提示 "statically-known trip count"。

### Phase 2 —— Var 别名与 tuple 解包

```python
with pl.manual_scope():
    a = self.k1(x)
    c = a                          # 普通 Var 别名
    p, q = self.kpair(x)           # tuple 解包
    d = self.k2(x, deps=[c, p])    # deps 引用了别名与解包元素
```

pass 合成：

```python
a__tid    = self.system.task_id_of(a)
c__tid    = a__tid                  # 别名转发 producer 的 task id
kpair_tmp = self.kpair(x)           # tuple 值
kpair_tmp__tid = self.system.task_id_of(kpair_tmp)
p__tid    = kpair_tmp__tid          # tuple 解包共享 producer 的 task id
q__tid    = kpair_tmp__tid
d = self.k2(x, deps=[c__tid, p__tid])
```

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

**Phase 1 实现**：`src/ir/transforms/derive_call_directions_pass.cpp`

- `PriorWriterCollector` —— 每作用域的第一写入者分析（自底向上缓存 + 自顶向下扫描）
- `CallDirectionMutator` —— 一个 `IRMutator`，用解析后的 `arg_directions` 向量重写每个非 builtin `Call`
- 复用 `include/pypto/codegen/orchestration/orchestration_analysis.h` 中的 `BufferRootCollector` 与 `ComputeGroupEffectiveDirections`

**Phase 2 实现**：`src/ir/transforms/utils/lower_manual_deps_to_task_id.cpp`（头文件 `include/pypto/ir/transforms/utils/lower_manual_deps_to_task_id.h`）

- 公开入口 `LowerManualDepsToTaskId(StmtPtr body) -> StmtPtr` —— `DeriveCallDirections` 在 Phase 1 完成 arg directions 后对每个函数 body 调用一次。当 body 中没有可达的 `RuntimeScopeStmt(manual=true)` 时直接返回原 body。
- `ManualDepResolveMutator`、`TaskRelevantVarCollector`、`PreallocateTaskIdVars`、`TaskIdLoweringMutator` —— 上文 Phase 2 描述的四个顺序子阶段。

**属性验证器**：`src/ir/verifier/verify_call_directions.cpp`（工厂函数声明在 `include/pypto/ir/verifier/verifier.h`）

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("derive_call_directions", &pass::DeriveCallDirections,
           "Derive Call attrs['arg_directions'] from callee param directions and buffer lineage, "
           "then lower user-declared deps=[...] edges inside manual scopes to TaskId companions.");
```

**类型存根**：`python/pypto/pypto_core/passes.pyi`

**手写 IR 辅助**：`python/pypto/ir/directions.py`（`make_call`、小写别名）—— 用于在 pass 运行前为 IR 片段附加显式方向的测试和手写 IR 代码。

**测试**：`tests/ut/ir/transforms/test_derive_call_directions.py`

- `TestDeriveDirectionMatrix` —— 为 (callee_dir, origin) → ArgDirection 映射表的每个单元各设一个测试，包含 R-seq（`pl.range`、`while`）与 R-prior（顶层 + 分支 / 顶层后跟 parallel）等边界情况
- `TestDeriveIdempotent` —— 两次运行该 pass 产生结构相等的 IR
- `TestDerivePreservesExplicit` —— 预先填充的 `arg_directions` 不会被覆盖
- `TestVerifyPositive` / `TestVerifyNegative` —— `CallDirectionsResolved` 属性验证器接受 pass 输出，并拒绝格式错误的 `arg_directions` 赋值

Phase 2 的覆盖测试位于 `tests/ut/ir/transforms/` 下的 manual scope 降级用例（单条 dep、多条 dep、循环 carry、Var 别名、tuple 解包）。
