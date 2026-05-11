# DeriveManualScopeDeps Pass

将 `with pl.manual_scope():` 区域内用户声明的 `deps=[...]` 边降级为基于 TaskId 的依赖图，供 orchestration codegen 生成 `params.add_dep(<task_id>)` 调用。

## 概述

PyPTO 的 orchestrator 依赖跟踪有两种 scope 形式：

- **Auto scope**（默认 `PTO2_SCOPE()`）：runtime 基于缓冲区读写重叠（OverlapMap）自动追踪。
- **Manual scope**（`with pl.manual_scope():` → `RuntimeScopeStmt(manual=true)`）：用户完全接管排序。runtime 跳过 OverlapMap；**所有需要的依赖边都必须由用户通过 `kernel(..., deps=[var, ...])` 显式声明**。本 pass 不再做任何数据流自动推导——之前的自动推导路径已被移除，原因是：当不同 kernel 在同一块 buffer 上原地复用时，数据流推导会产生大量伪边。

Python 侧 (`passes.derive_manual_scope_deps()`) 和 pipeline 中的 pass 名仍叫 `DeriveManualScopeDeps`；C++ 实现是 `LowerManualDepsToTaskId` lowering。两者指同一个 pass 槽位——pass 名为保持 binding API 兼容性而保留。

**何时使用**：在 `DeriveCallDirections`（解析 per-arg `ArgDirection`，产出 `CallDirectionsResolved`）之后、尾部 `Simplify` 之前运行。`Default` 策略中是第 33 个 pipeline pass。

## 属性

| Required | Produced | Invalidated |
| -------- | -------- | ----------- |
| `CallDirectionsResolved` | — | — |

未声明可验证的后置条件。本 pass 对已降级的 IR 在结构上是幂等的：第二次运行会找到相同的 `kAttrUserManualDepEdges` 条目、解析出相同的闭包、并通过 `tid_map_` 按指针等同性复用同一批 TaskId 配套变量。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::DeriveManualScopeDeps()` | `passes.derive_manual_scope_deps()` | Program 级 |

```cpp
Pass DeriveManualScopeDeps();
```

```python
from pypto.pypto_core import passes
pass_obj = passes.derive_manual_scope_deps()
program_after = pass_obj(program)
```

## 算法

对每个含 manual scope 的函数 body，`LowerOneFunction` 依次运行四个子 pass。

### 阶段 1 — `ManualDepResolveMutator`

对 manual scope 内的每个 kernel `Call`，把 `Call.attrs["user_manual_dep_edges"]`（parser 在 DSL 传 `deps=[var, ...]` 时写入的 Tensor Var 边）复制到 `Call.attrs["manual_dep_edges"]`。不做任何自动推导。单 call 的边数与 `kManualDepEdgeLimit = 16`（对应 runtime `PTO2_MAX_EXPLICIT_DEPS`）做检查，超限时抛出携带 call span 的 internal error。

### 阶段 2 — `TaskRelevantVarCollector`（闭包分析）

从每个 `kAttrManualDepEdges` 集合中的 Tensor Var 出发，沿以下关系传播"需要 TaskId 配套"标记：

- **Var 别名**（`b = a` AssignStmt 和 `b = tuple[i]` TupleGetItem 解包）。
- **`ForStmt.iter_args` ↔ `init_value`**（init 来自被标记 Var 的 iter_arg 也需要 TaskId carry，反之亦然）。
- **`ForStmt.return_vars` ↔ `iter_args`**（TaskId carry iter_arg 产出的 rv 本身也是 TaskId carry）。
- **`YieldStmt` source ↔ destination**（双向。两个方向都必要：`deps=[<iter_arg>]` 走 dest→src，而 `deps=[<kernel_lhs>]` 走 src→dest 抵达 carry 目的地）。

不动点闭包结束后形成三个集合：`needs_tid_`（需要配套的 Var）、`kernel_lhs_`（user kernel Call 的 LHS——将走 `system.task_id_of` 合成路径）、`import_vars_`（在 `needs_tid_` 中但没有 AssignStmt 定义的 Var，通常是被用作 iter_arg init 的函数参数）。

### 阶段 3 — `PreallocateTaskIdVars`

为 `needs_tid_` 中每个 Var 预分配 TaskId 配套：

- 普通 `Var`（非 IterArg，如 kernel LHS 或函数参数）→ 一个新 `Var`，name_hint 为 `<原名>__tid`，类型 `ScalarType(DataType::TASK_ID)`。
- `IterArg` → 一个新 `IterArg`，相同后缀；其 init 指向外层 Var 的配套（在已建立的 `tid_map_` 中查到）。嵌套循环时，内层 iter_arg 需要外层配套先到位，因此 IterArg 分配阶段做不动点扫描：init 配套尚未分配的 iter_arg 会被反复重试直到链上收敛。

`tid_map_: const Var* → VarPtr` 是配套身份的唯一来源；其他阶段都通过它查询，避免指针漂移。

### 阶段 4 — `TaskIdLoweringMutator`（IR 改写）

一次 IRMutator pass 完成所有 TaskId 基础设施的注入：

- 对每个 LHS 在 `needs_tid_` 中的 kernel `Call` AssignStmt，紧随其后注入 `<lhs>__tid = system.task_id_of(<lhs>)`。
- 对每个 LHS 在 `needs_tid_` 中的 `tensor.create` AssignStmt（无 producer 的占位 buffer），注入 `<lhs>__tid = system.task_invalid()`。
- 对每个普通 Var 别名 AssignStmt（`b = a`），注入 `b__tid = a__tid`。
- 对每个 TupleGetItem AssignStmt（`b = tuple[i]`），注入 `b__tid = tuple_var__tid`（所有解包元素共用 tuple-producing call 的 task id）。
- 对每个 kernel `Call`：把 `kAttrManualDepEdges` 中的 Tensor Var 改写为 TaskId 配套；并附上 `kAttrTaskIdVar` 指向 LHS 的配套（让后续 sibling 可通过该 attr 解析 `deps=[lhs]` 而不必重跑闭包）。
- 对 manual scope 内每个 `ForStmt`：为每个落在 `needs_tid_` 中的现有 iter_arg 追加一个 TaskId iter_arg + return_var 配套；yield 值列表对称扩展。
- 对 `import_vars_`（作为 TaskId iter_arg 种子的函数参数），在函数 body 入口前置 `<param>__tid = system.task_invalid()` AssignStmt，让配套有 SSA 定义供 codegen 引用。

kernel-Call 改写后，`kAttrManualDepEdges` 内保存的是**降级后**形式（TaskId Var）。codegen 消费这个 attr；原始 Tensor Var 形式保留在 `kAttrUserManualDepEdges` 中供圆训打印。

## 示例

### 单条依赖边

```python
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64], pl.FP32],
         out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
    with pl.manual_scope():
        scratch = self.stage1(x, scratch)
        out = self.stage2(scratch, out, deps=[scratch])
    return out
```

pass 运行后：

```python
scratch__ssa_v0__tid: pl.Scalar[pl.TASK_ID] = self.system.task_invalid()  # import var 种子
with pl.manual_scope():
    scratch__ssa_v5 = self.stage1(x, scratch)
    scratch__ssa_v5__tid: pl.Scalar[pl.TASK_ID] = self.system.task_id_of(scratch__ssa_v5)
    out__ssa_v7 = self.stage2(scratch__ssa_v5, out, deps=[scratch__ssa_v5__tid])
```

codegen 从改写后的 dep edge 发出 `params_t1.add_dep(scratch__ssa_v5__tid);`。

### 多依赖 + 循环 carry（case1 形态）

```python
with pl.manual_scope():
    for phase in pl.range(N_PHASES):
        for branch in pl.parallel(N_BRANCHES):
            row = (phase * N_BRANCHES + branch) * TILE_M
            out = self.kernel_stripe(data, row, 1.0, out, deps=[out])
```

`out` 在每次循环都被重新绑定，因此两个 ForStmt 上都会追加 TaskId iter_arg（承载上一轮的 task id）。pass 运行后：

```python
for phase, (out__iter_v1, out__iter_v1__tid) in pl.range(4, init_values=(out, out__ssa_v0__tid)):
    for branch, (out__iter_v3, out__iter_v3__tid) in pl.parallel(4, init_values=(out__iter_v1, out__iter_v1__tid)):
        out__ssa_v5 = self.kernel_stripe(..., deps=[out__iter_v3__tid])
        out__ssa_v5__tid = self.system.task_id_of(out__ssa_v5)
        out__rv_v4, out__rv_v4__tid = pl.yield_(out__ssa_v5, out__ssa_v5__tid)
    out__rv_v2, out__rv_v2__tid = pl.yield_(out__rv_v4, out__rv_v4__tid)
```

orchestration codegen 把 `pl.parallel` 上的 TaskId iter_arg 视作**大小为 `N_BRANCHES` 的数组 carry**（仅当 trip count 为编译期常量时）：分配 `PTO2TaskId arr[N_BRANCHES]`，每个 parallel iter 写入自己的槽位；下游消费者对每个槽各发一次 `add_dep`。这就保证 phase fence 真正等待**全部** parallel iter，而不是只等"最后被发射"的那个 task。大小上限同样是 `PTO2_MAX_EXPLICIT_DEPS = 16`；超过会在 codegen 时报清晰错误。带 manual dep 的 `pl.parallel` 若 trip count 不是常量，codegen 会直接报"statically-known trip count"错误。

### Var 别名与元组解包

```python
with pl.manual_scope():
    a = self.k1(x)
    c = a                          # 普通 Var 别名
    p, q = self.kpair(x)           # 元组解包
    d = self.k2(x, deps=[c, p])    # deps 引用别名和解包元素
```

pass 合成的代码：

```python
a__tid    = self.system.task_id_of(a)
c__tid    = a__tid                  # 别名转发 producer 的 task id
kpair_tmp = self.kpair(x)
kpair_tmp__tid = self.system.task_id_of(kpair_tmp)
p__tid    = kpair_tmp__tid          # 解包元素共用 producer 的 task id
q__tid    = kpair_tmp__tid
d = self.k2(x, deps=[c__tid, p__tid])
```

## 参见

- [DeriveCallDirections（编号 32）](32-derive_call_directions.md)——必需前置 pass；负责解析 per-arg `ArgDirection`。
- [System ops: `task_invalid` / `task_id_of`](../ir/05-operators.md#syncop-synchronization-operations)——本 pass 合成的两个 builtin。
- [DataType: `TASK_ID`](../ir/02-types.md#scalartype)——用于 TaskId 配套的不透明 64-bit handle。
- [Orchestration codegen: manual scope + array carry](../codegen/01-orchestration_codegen.md)——降级后 IR 如何被发射。
- [Python syntax: manual scope 与 deps](../language/00-python_syntax.md#manual-dependency-primitives)——表层语义。
