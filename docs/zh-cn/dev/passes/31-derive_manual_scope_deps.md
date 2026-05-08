# DeriveManualScopeDeps Pass

为 `with pl.manual_scope():` 区域内的 kernel `Call` 解析任务依赖边，并将其写入 `Call.attrs["manual_dep_edges"]` 供 codegen 使用。

## 概述

PyPTO 的 orchestrator 依赖跟踪有两种 scope 形式：

- **Auto scope**（默认 `PTO2_SCOPE()` 区域）：runtime 基于缓冲区读写重叠（OverlapMap）自动推导任务依赖。
- **Manual scope**（`with pl.manual_scope():` → `RuntimeScopeStmt(manual=true)`）：用户接管排序。runtime 在该区域内跳过 OverlapMap，codegen 必须改为发出显式 `params.add_dep(task_<m>);` 调用。

`DeriveManualScopeDeps` 是连接 DSL 意图与 runtime 语义的 pass。对 manual scope 内每个 kernel `Call`，它计算以下两类边的并集：

1. **用户提供的边**——由 DSL `kernel(..., deps=[var, ...])` kwarg 传入的所有 var，由 parser 写到 `Call.attrs["user_manual_dep_edges"]`。
2. **数据流边**——每个 `ArgDirection` 不为 `NoDep` 且其 `Var` 解析到当前 manual scope 内 producer（先前 kernel `AssignStmt` 的 LHS）的 tensor 参数。

解析结果写入 `Call.attrs["manual_dep_edges"]`（`std::vector<VarPtr>`），由 orchestration codegen 读取并为每条边发出 `params_t<n>.add_dep(task_<m>);`。该列表上限为 16 条以匹配 runtime `PTO2_MAX_EXPLICIT_DEPS`；超过上限会以指向出错调用点的内部错误终止。

**何时使用**：在 `DeriveCallDirections` 之后运行（数据流分析需要已解析的 per-arg direction 来正确处理 `NoDep` 槽位），并在最终 `Simplify` 之前。在 `Default` 策略中是第 31 个 pipeline pass，介于 `DeriveCallDirections`（编号 30）与末尾 `Simplify`（utility 编号 91）之间。

## 属性

| Required | Produced | Invalidated |
| -------- | -------- | ----------- |
| `CallDirectionsResolved` | — | — |

本 pass 暂未声明可验证的后置条件；它在结构上是幂等的——重复运行会写入相同的 `manual_dep_edges` 集合。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::DeriveManualScopeDeps()` | `passes.derive_manual_scope_deps()` | Program 级 |

**工厂函数**：

```cpp
Pass DeriveManualScopeDeps();
```

**Python 用法**：

```python
from pypto.pypto_core import passes

pass_obj = passes.derive_manual_scope_deps()
program_with_edges = pass_obj(program)
```

## 算法

本 pass 是一个 `ProgramPass`，对每个函数 body 运行一遍 `ManualDepMutator`。

### 1. Manual scope 跟踪

mutator 用整型 `manual_depth_` 计数器记录嵌套深度：进入 `RuntimeScopeStmt(manual=true)` 时 +1，退出时 -1。`manual_depth_ == 0` 时（不在 manual scope 内），mutator 对 `AssignStmt` / `EvalStmt` 的访问是 no-op。

进入 manual scope 时还会**保存并清空** producer-Var 映射（见步骤 2），退出时恢复。这样可隔离每个 scope：scope A 内的 producer 不能被 scope B 引用为 manual dep edge。

### 2. Producer-Var 映射

manual scope 内，每个 RHS 为非 builtin kernel `Call` 的 `AssignStmt` 都会把 LHS `Var*` 注册到 `producer_map_`。Builtin op（`tensor.*` / `tile.*` / `system.*`）永远不是 producer，因为它们不会触发 runtime 任务提交。

### 3. 单 call 的边解析

对 manual scope 内每个 kernel `Call`（无论作为 `AssignStmt` 的 RHS 还是 `EvalStmt` 的表达式），辅助函数 `ResolveManualDepsForCall` 分两阶段收集边并按 `Var*` 去重：

1. **用户提供**——扫描 `Call.attrs["user_manual_dep_edges"]`（DSL 传 `deps=[var]` 时由 parser 写入），保留顺序。
2. **数据流**——遍历每个位置参数，若其 `ArgDirection` 不为 `NoDep` 且 `AsVarLike(arg)` 解析到的 `Var*` 在 `producer_map_` 中，将该 producer 追加进结果。

结果写回为 `Call.attrs["manual_dep_edges"]`（一个新的 `std::vector<VarPtr>`）。上限检查通过 `INTERNAL_CHECK_SPAN(deps.size() <= 16, call->span_)` 完成，错误信息携带源码位置。

### 4. Var 类型 attr 在 remap 中保持同步

由于 edge attr 中保存的 `VarPtr` 必须与定义点保持一致，本次改动同步扩展了基类 `IRMutator::VisitExpr_(Call)` 与 `IRVisitor::VisitExpr_(Call)`，让它们通过标准 `var_remap_` / use-site visit 路径处理 `kAttrManualDepEdges` / `kAttrUserManualDepEdges` 中的 Var；`ConvertToSSA` 用其自身的 `cur_` 映射做了同样的事情。

## 示例

### 数据流自动推导

```python
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    with pl.manual_scope():
        a = self.k1(x)
        b = self.k2(a)        # k2 读 `a` -> 自动推出边 task_0
    return b
```

pass 运行后，`k2` 调用携带 `attrs["manual_dep_edges"] = [a]`。codegen 输出：

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    Arg params_t0;
    params_t0.add_input(ext_x);
    TaskOutputTensors task_0_outs = rt_submit_aiv_task(0, params_t0);
    PTO2TaskId task_0 = task_0_outs.task_id();

    Arg params_t1;
    params_t1.add_input(/* a 的别名 */);
    params_t1.add_dep(task_0);          // <-- 来自 manual_dep_edges
    TaskOutputTensors task_1_outs = rt_submit_aiv_task(1, params_t1);
    PTO2TaskId task_1 = task_1_outs.task_id();
}
```

### 显式用户依赖

`deps=[var, ...]` 添加数据流不可见的边：

```python
with pl.manual_scope():
    a = self.k1(x)
    b = self.k2(x)
    c = self.k3(x, deps=[a, b])    # c 只读 x，但排序需依赖 a, b
```

pass 运行后，`c` 携带 `manual_dep_edges = [a, b]`；codegen 同时发出 `params_t2.add_dep(task_0)` 与 `params_t2.add_dep(task_1)`。

### 用 `pl.no_dep` 抑制自动边

`pl.no_dep(arg)` 标记（Phase 2）在 parse 时把对应槽位的 `ArgDirection` 设为 `NoDep`；本 pass 在数据流扫描中跳过该槽位，因此其背后的 producer 不会成为 manual dep edge。

## 参见

- [DeriveCallDirections（编号 30）](30-derive_call_directions.md)——必需前置 pass；负责把 `pl.no_dep(...)` 解析为 `NoDep` direction。
- [IR hierarchy: ScopeStmt](../ir/01-hierarchy.md#scopestmt-details)——`RuntimeScopeStmt` 与 auto/manual 标志。
- [Python syntax: scope context managers](../language/00-python_syntax.md#scope-context-managers)——`with pl.manual_scope():` 表层语法。
